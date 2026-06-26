# EEG-only Transformer Release

This folder contains the EEG-only diffusion model with a Transformer-based EEG encoder,
plus training, sampling, and k-fold training scripts.

## Files
- model_128_eegonly_transformer.py: EEG-only diffusion model with Conv+Transformer EEG encoder.
- train_subject_128_group3_eegonly_transformer.py: Subject-wise training (3 class groups).
- sample_subject_all_group_128_eegonly_transformer.py: Sampling for a subject/group.
- run_sample_subject_group_range_eegonly_transformer.py: Range runner for multi-subject/group sampling.
- train_subject_128_group3_eegonly_transformer_kfold.py: K-fold training for a subject/group.

## Example Commands
Training:
```bash
python train_subject_128_group3_eegonly_transformer.py --data_root ./preproc_data --subject_ids 1,2,3 --epochs 200 --batch_size 16 --img_size 128 --base_channels 128 --ch_mult 1,2,4,8 --n_res_blocks 5 --num_timesteps 500 --eeg_tf_heads 4 --eeg_tf_layers 2
``

Sampling:
```bash
python sample_subject_all_group_128_eegonly_transformer.py --data_root ./preproc_data --subject_id 1 --group_id 1 --img_size 128 --sample_steps 500
```

Range sampling:
```bash
python run_sample_subject_group_range_eegonly_transformer.py --data_root ./preproc_data --subjects 1-5 --groups 1-3 --img_size 128 --sample_steps 500
```

K-fold (all folds):
```bash
python train_subject_128_group3_eegonly_transformer_kfold.py --data_root ./preproc_data --subject_id 1 --group_id 1 --k_folds 5
```