"""
rbln_compile_diagnostics.py
───────────────────────────────────────────────────────────────────
RBLN graph conversion diagnostic for the SupCon EEG encoder.

Tests 6 compile targets in order, from simplest to full encoder.
Saves per-target logs and a markdown report.

Run on NPU server only — requires rebel SDK.
Usage:
    python rbln_compile_diagnostics.py [--ckpt <path>] [--sid 24]

Outputs:
    outputs/logs/rbln_compile_diagnostics/<target>_{success,failure}.log
    outputs/reports/rbln_compile_diagnostic_report.md
"""

import argparse
import math
import os
import sys
import time
import traceback

import torch
import torch.nn as nn

# ── rebel import (graceful stub if not available) ──────────────────────────
try:
    import rebel
    REBEL_AVAILABLE = True
except ImportError:
    REBEL_AVAILABLE = False
    print("[WARN] rebel not found — compile steps will be skipped, eager-only mode.")

# ── local model imports ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from model_128_eegonly_transformer import (
    OccipitalChannelGate,
    MultiScaleStem,
    EEGEncoderV2,
)
from model_eeg_dino import EEGDINORegressor

# ── constants ──────────────────────────────────────────────────────────────
INPUT_SHAPE   = (1, 32, 2048)   # (B, C, T) — fixed batch=1 for NPU compile
HIDDEN_DIM    = 256
N_HEADS       = 4
N_LAYERS      = 4               # Exp26 checkpoint uses 4 layers
OUT_DIM       = 256

LOG_DIR    = "outputs/logs/rbln_compile_diagnostics"
REPORT_DIR = "outputs/reports"
os.makedirs(LOG_DIR,    exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Compile target wrappers
# ═══════════════════════════════════════════════════════════════════════════

class Target1_ConvFrontend(nn.Module):
    """OccipitalChannelGate + MultiScaleStem + proj Conv1d.
    Input : (1, 32, 2048)
    Output: (1, HIDDEN_DIM, 1024)
    """
    def __init__(self):
        super().__init__()
        self.channel_gate = OccipitalChannelGate(32)
        self.stem = MultiScaleStem(32, n_filters=32, dropout=0.0)
        self.proj = nn.Sequential(
            nn.Conv1d(96, HIDDEN_DIM, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, HIDDEN_DIM),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)
        x = self.channel_gate(x)
        h = self.stem(x)
        return self.proj(h)


class Target2_SinPEStatic(nn.Module):
    """Sinusoidal PE with STATIC seq_len (pre-computed, avoids torch.arange at runtime).
    Input : (1, 1024, HIDDEN_DIM)   (after conv, transposed)
    Output: (1, 1024, HIDDEN_DIM)
    """
    SEQ_LEN = 1024  # 2048 // 2

    def __init__(self):
        super().__init__()
        # pre-compute and register as buffer (static, not dynamic)
        pe = self._make_pe(self.SEQ_LEN, HIDDEN_DIM)
        self.register_buffer("pe", pe)          # (1, SEQ_LEN, HIDDEN_DIM)

    @staticmethod
    def _make_pe(seq_len: int, dim: int) -> torch.Tensor:
        pos = torch.arange(seq_len).unsqueeze(1).float()
        i   = torch.arange(0, dim, 2).float()
        div = torch.exp(-i * (math.log(10000.0) / dim))
        emb = torch.zeros(seq_len, dim)
        emb[:, 0::2] = torch.sin(pos * div)
        emb[:, 1::2] = torch.cos(pos * div[:dim // 2])
        return emb.unsqueeze(0)                 # (1, seq_len, dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.pe


class Target3_OneTransformerLayer(nn.Module):
    """Single TransformerEncoderLayer (batch_first=True).
    Input : (1, 1024, HIDDEN_DIM)
    Output: (1, 1024, HIDDEN_DIM)
    """
    def __init__(self):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM, nhead=N_HEADS,
            dim_feedforward=HIDDEN_DIM * 4,
            dropout=0.0, batch_first=True,
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.layer(h)


class Target4_FourTransformerLayers(nn.Module):
    """Four stacked TransformerEncoderLayers.
    Input : (1, 1024, HIDDEN_DIM)
    Output: (1, HIDDEN_DIM)   (mean-pooled)
    """
    def __init__(self):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=HIDDEN_DIM, nhead=N_HEADS,
                dim_feedforward=HIDDEN_DIM * 4,
                dropout=0.0, batch_first=True,
            ),
            num_layers=N_LAYERS,
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.transformer(h).mean(1)


class Target5_EncoderV2StaticPE(nn.Module):
    """EEGEncoderV2 with dynamic sinusoidal PE replaced by static buffer.
    Input : (1, 32, 2048)
    Output: (1, OUT_DIM)
    This is the RBLN-compatible rewrite candidate.
    """
    SEQ_LEN = 1024  # 2048 // 2

    def __init__(self):
        super().__init__()
        self.channel_gate = OccipitalChannelGate(32)
        self.stem = MultiScaleStem(32, n_filters=32, dropout=0.0)
        self.proj = nn.Sequential(
            nn.Conv1d(96, HIDDEN_DIM, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, HIDDEN_DIM),
            nn.SiLU(),
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=HIDDEN_DIM, nhead=N_HEADS,
                dim_feedforward=HIDDEN_DIM * 4,
                dropout=0.0, batch_first=True,
            ),
            num_layers=N_LAYERS,
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, OUT_DIM),
        )
        # static PE buffer
        pe = Target2_SinPEStatic._make_pe(self.SEQ_LEN, HIDDEN_DIM)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)
        x = self.channel_gate(x)
        h = self.stem(x)
        h = self.proj(h)
        h = h.transpose(1, 2)          # (B, SEQ_LEN, HIDDEN_DIM)
        h = h + self.pe
        h = self.transformer(h).mean(1)
        return self.fc(h)


class Target6_FullEncoderOriginal(nn.Module):
    """Original EEGEncoderV2 with dynamic sinusoidal PE (expected to fail).
    Input : (1, 32, 2048)
    Output: (1, OUT_DIM)
    """
    def __init__(self):
        super().__init__()
        self.enc = EEGEncoderV2(
            eeg_channels=32,
            eeg_hidden_dim=HIDDEN_DIM,
            out_dim=OUT_DIM,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            dropout=0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostic runner
# ═══════════════════════════════════════════════════════════════════════════

TARGETS = [
    ("T1_conv_frontend",           Target1_ConvFrontend,         (1, 32, 2048),     "float32"),
    ("T2_sinpe_static_buffer",     Target2_SinPEStatic,          (1, 1024, HIDDEN_DIM), "float32"),
    ("T3_one_transformer_layer",   Target3_OneTransformerLayer,  (1, 1024, HIDDEN_DIM), "float32"),
    ("T4_four_transformer_layers", Target4_FourTransformerLayers,(1, 1024, HIDDEN_DIM), "float32"),
    ("T5_encoder_static_pe",       Target5_EncoderV2StaticPE,    (1, 32, 2048),     "float32"),
    ("T6_full_encoder_original",   Target6_FullEncoderOriginal,  (1, 32, 2048),     "float32"),
]


def run_eager(model: nn.Module, input_shape: tuple, device: torch.device):
    """Run one forward pass in eager mode. Returns (output_shape_str, error_or_None)."""
    model.eval()
    dummy = torch.randn(*input_shape, device=device)
    try:
        with torch.no_grad():
            out = model(dummy)
        return str(tuple(out.shape)), None
    except Exception as e:
        return None, traceback.format_exc()


def run_compile(model: nn.Module, name: str, input_shape: tuple, dtype: str):
    """Try rebel.compile_from_torch. Returns (success: bool, error_or_None)."""
    if not REBEL_AVAILABLE:
        return None, "rebel not installed"
    try:
        compiled = rebel.compile_from_torch(
            model,
            input_info=[(name, list(input_shape), dtype)],
        )
        return True, None
    except Exception as e:
        return False, traceback.format_exc()


def save_log(name: str, content: str, suffix: str):
    path = os.path.join(LOG_DIR, f"{name}_{suffix}.log")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   default=None, help="Optional: load checkpoint weights for T5/T6")
    parser.add_argument("--sid",    type=int, default=24)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}  |  rebel: {REBEL_AVAILABLE}\n")

    results = []   # list of dicts for report

    for (name, cls, input_shape, dtype) in TARGETS:
        print(f"{'='*60}")
        print(f"Target: {name}  input={input_shape}")

        model = cls().to(device)
        model.eval()

        # ── eager forward ──────────────────────────────────────────
        t0 = time.time()
        out_shape, eager_err = run_eager(model, input_shape, device)
        eager_ms = (time.time() - t0) * 1000

        if eager_err:
            print(f"  [EAGER] FAIL")
            save_log(name, eager_err, "eager_failure")
            results.append({
                "name": name, "input": str(input_shape),
                "eager": "FAIL", "eager_out": "-",
                "compile": "SKIP", "compile_note": "eager failed",
            })
            continue
        else:
            print(f"  [EAGER] OK  out={out_shape}  {eager_ms:.1f}ms")
            save_log(name, f"OK  output={out_shape}  {eager_ms:.1f}ms", "eager_success")

        # ── rebel compile ──────────────────────────────────────────
        if REBEL_AVAILABLE:
            print(f"  [COMPILE] attempting rebel.compile_from_torch ...")
            t0 = time.time()
            ok, compile_err = run_compile(model, "x", input_shape, dtype)
            compile_ms = (time.time() - t0) * 1000

            if ok:
                print(f"  [COMPILE] SUCCESS  {compile_ms:.0f}ms")
                save_log(name, f"SUCCESS  {compile_ms:.0f}ms", "compile_success")
                compile_status = "SUCCESS"
                compile_note   = f"{compile_ms:.0f}ms"
            else:
                # extract first meaningful line from traceback
                first_err = compile_err.strip().split("\n")[-1][:120]
                print(f"  [COMPILE] FAIL — {first_err}")
                save_log(name, compile_err, "compile_failure")
                compile_status = "FAIL"
                compile_note   = first_err
        else:
            compile_status = "SKIP"
            compile_note   = "rebel not installed"

        results.append({
            "name": name, "input": str(input_shape),
            "eager": "OK", "eager_out": out_shape,
            "compile": compile_status, "compile_note": compile_note,
        })

    # ── Markdown report ────────────────────────────────────────────────────
    lines = [
        "# RBLN Compile Diagnostic Report",
        f"\n**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Device**: {device}  |  **rebel available**: {REBEL_AVAILABLE}",
        f"**Input shape**: {INPUT_SHAPE}  (B=1, C=32, T=2048)\n",
        "## Results\n",
        "| Target | Input | Eager | Eager output | Compile | Note |",
        "|--------|-------|-------|-------------|---------|------|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['input']} | {r['eager']} | "
            f"{r['eager_out']} | {r['compile']} | {r['compile_note'][:80]} |"
        )

    lines += [
        "\n## Target Descriptions\n",
        "| Target | Description | RBLN risk |",
        "|--------|-------------|-----------|",
        "| T1_conv_frontend           | OccipitalChannelGate + MultiScaleStem + proj Conv1d | low |",
        "| T2_sinpe_static_buffer     | Pre-computed sinusoidal PE as buffer (no torch.arange at runtime) | low |",
        "| T3_one_transformer_layer   | Single TransformerEncoderLayer (batch_first=True) | medium |",
        "| T4_four_transformer_layers | 4× TransformerEncoderLayer + mean pool | high |",
        "| T5_encoder_static_pe       | Full encoder with dynamic PE → static buffer (fix candidate) | high |",
        "| T6_full_encoder_original   | Original EEGEncoderV2 with dynamic torch.arange PE | expected fail |",
        "\n## Known RBLN Risk Factors\n",
        "- `torch.arange(seq_len, device=device)` inside `forward` → dynamic tensor creation",
        "- `nn.TransformerEncoderLayer(batch_first=True)` → scatter/gather attention ops",
        "- `h.size(1)` used to derive positional embedding shape at runtime",
        "\n## Recommended Fix if T3/T4 fail\n",
        "If `TransformerEncoderLayer` itself fails RBLN compilation:",
        "- **Option A**: Replace 4-layer transformer with Conv1D temporal aggregator (compile-safe)",
        "- **Option B**: Run encoder on CPU, compile only downstream fusion MLP + projector on NPU",
        "- **Option C**: Pre-compute EEG latents offline; deploy latent→image pipeline only on NPU",
        "",
        "If only dynamic PE fails (T6 fails, T5 succeeds):",
        "- Apply static PE buffer patch to `EEGEncoderV2._sinusoidal_pos_embed`",
        "- This is a minimal, non-breaking fix",
        "",
        f"Log directory: `{LOG_DIR}`",
    ]

    report_path = os.path.join(REPORT_DIR, "rbln_compile_diagnostic_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n{'='*60}")
    print(f"Report saved: {report_path}")
    print(f"Logs saved:   {LOG_DIR}/")
    print("\nSummary:")
    for r in results:
        status = f"eager={r['eager']:4s}  compile={r['compile']}"
        print(f"  {r['name']:<40s}  {status}")


if __name__ == "__main__":
    main()
