# VI-primary gated residual VS→VI

The raw+TF VI-only checkpoint is the primary model. A frozen raw+TF VS encoder
contributes an identity-adapted residual through a scalar gate initialized to
zero. Epoch 0 is therefore exactly the existing VI-only model and is included
in validation selection.

The fixed S24 development protocol trains the adapter and gate alone for 10
epochs, then jointly fine-tunes the VI student at one tenth of the adapter/head
learning rate. Selection is VI validation BAC, Top-3, then Top-5 with metrics
rounded to 12 decimals. The selected model is evaluated on VI test once.

```bash
python -u run_vi_rawtf_gated_residual.py --stage audit --subjects 24 --seed 42
python -u run_vi_rawtf_gated_residual.py --stage run --subject_id 24 --subjects 24 --seed 42 --fp16
python -u run_vi_rawtf_gated_residual.py --stage summary --subjects 24 --seed 42
```
