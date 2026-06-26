# RBLN Compile Diagnostic Report

**Date**: 2026-06-27 01:54:26
**Device**: cpu  |  **rebel available**: False
**Input shape**: (1, 32, 2048)  (B=1, C=32, T=2048)

## Results

| Target | Input | Eager | Eager output | Compile | Note |
|--------|-------|-------|-------------|---------|------|
| T5_encoder_static_pe | (1, 32, 2048) | OK | (1, 256) | SKIP | rebel not installed |
| T6_full_encoder_original | (1, 32, 2048) | OK | (1, 256) | SKIP | rebel not installed |
| T7_s24_staticpe_real_ckpt | (1, 32, 2048) | OK | (1, 256) | SKIP | rebel not installed |

## Track B Known Results (NPU server, 2026-06-27)

| Component | Detail | Status |
|-----------|--------|--------|
| checkpoint_loading | S01/S18/S24 SupCon clean load | SUCCESS |
| cpu_encoder_inference | output shape=(1, 256) | SUCCESS |
| rbln_smi_device_detect | RBLN-CA22 x2 recognized | SUCCESS |
| graph_conversion_transformer | TransformerEncoder TVM convert | FAIL |
| full_encoder_compile | EEGEncoderV2 rebel.compile | FAIL |
| npu_inference | blocked by compile failure | BLOCKED |

## Target Descriptions

| Target | Description | RBLN risk |
|--------|-------------|-----------|
| T1_conv_frontend           | OccipitalChannelGate + MultiScaleStem + proj Conv1d | low |
| T2_sinpe_static_buffer     | Pre-computed sinusoidal PE as buffer (no torch.arange at runtime) | low |
| T3_one_transformer_layer   | Single TransformerEncoderLayer (batch_first=True) | medium |
| T4_four_transformer_layers | 4x TransformerEncoderLayer + mean pool | high |
| T5_encoder_static_pe       | Full encoder with dynamic PE -> static buffer (fix candidate) | high |
| T6_full_encoder_original   | Original EEGEncoderV2 with dynamic torch.arange PE | expected fail |
| T7_s24_staticpe_real_ckpt  | Static-PE encoder loaded with actual S24 SupCon weights | high |

## Known RBLN Risk Factors

- `torch.arange(seq_len, device=device)` inside `forward` → dynamic tensor creation
- `nn.TransformerEncoderLayer(batch_first=True)` → scatter/gather attention ops
- `h.size(1)` used to derive positional embedding shape at runtime

## Recommended Fix if T3/T4 fail

If `TransformerEncoderLayer` itself fails RBLN compilation:
- **Option A**: Replace 4-layer transformer with Conv1D temporal aggregator (compile-safe)
- **Option B**: Run encoder on CPU, compile only downstream fusion MLP + projector on NPU
- **Option C**: Pre-compute EEG latents offline; deploy latent→image pipeline only on NPU

If only dynamic PE fails (T6 fails, T5 succeeds):
- Apply static PE buffer patch to `EEGEncoderV2._sinusoidal_pos_embed`
- This is a minimal, non-breaking fix

Log directory: `outputs/logs/rbln_compile_diagnostics`