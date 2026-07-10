# Exp43 full-test evaluation on Colab

This workflow generates one deterministic image for every VI test trial and
compares C0/C1 using identical trials and initial diffusion noise.

## Preflight

Run training/evaluation in the foreground. Before importing PEFT/diffusers:

```python
!pip uninstall -y torchao
```

Verify the corrected evaluation script is present after clone/pull:

```python
from pathlib import Path

repo = Path("/content/vsvi_project")
assert (repo / "eval_exp43_full_test.py").is_file()
assert (repo / "summarize_exp43_full_test.py").is_file()
```

## S24 C0

```python
import subprocess

common = [
    "--supcon_ckpt", "/content/vsvi_project/checkpoints_vsre_dino/20260604_091352_ch32_merged_ep200_supcon",
    "--data_root", "/content/vsvi_project/preproc_vi_re",
    "--img_root", "/content/vsvi_project/preproc_data_vi/images",
    "--subject_id", "24",
    "--lora_r", "32",
    "--n_eeg_tokens", "16",
    "--per_class_total", "60",
    "--ddim_steps", "30",
    "--batch_size", "2",
    "--seed", "20260711",
    "--split_seed", "42",
    "--out_root", "/content/drive/MyDrive/vsvi_data/gen_images_full",
]

c0_ckpt = (
    "/content/drive/MyDrive/vsvi_data/checkpoints_exp43_vi_lora/"
    "20260703_184527_exp43_vi_c0_s24_lora_r32_tok16_ep100_percls60/"
    "subj24_exp43_c0_lora_best.pt"
)

cmd = ["python", "-u", "/content/vsvi_project/eval_exp43_full_test.py", "--ckpt", c0_ckpt] + common
result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print(result.stdout)
assert result.returncode == 0
```

## S24 C1

Find the C1 checkpoint and run with the exact same `common` arguments:

```python
from pathlib import Path

root = Path("/content/drive/MyDrive/vsvi_data/checkpoints_exp43_vi_lora")
c1_matches = sorted(root.rglob("subj24_exp43_c1_lora_best.pt"))
assert c1_matches
c1_ckpt = str(c1_matches[-1])

cmd = ["python", "-u", "/content/vsvi_project/eval_exp43_full_test.py", "--ckpt", c1_ckpt] + common
result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print(result.stdout)
assert result.returncode == 0
```

Do not add `--overwrite` unless deliberately replacing an existing complete
evaluation. C0 and C1 must use identical seed, split seed, and dataset cap.

## Aggregate completed subjects

```python
!python -u /content/vsvi_project/summarize_exp43_full_test.py \
  --root /content/drive/MyDrive/vsvi_data/gen_images_full
```

Primary outputs:

```text
gen_images_full/S24/c0/manifest.csv
gen_images_full/S24/c0/metrics.json
gen_images_full/S24/c0/per_class_metrics.csv
gen_images_full/S24/c0/confusion_matrix.csv
gen_images_full/S24/c0/S24_exp43_c0_full_grid.png
gen_images_full/exp43_full_test_summary.csv
```
