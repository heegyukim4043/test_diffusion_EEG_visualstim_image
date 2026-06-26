# run_sample_and_xai_eegonly.py
import argparse
import subprocess
import sys


def build_sample_cmd(args):
    cmd = [
        sys.executable,
        "sample_subject_all_group_128_eegonly.py",
        "--data_root", args.data_root,
        "--subject_id", str(args.subject_id),
        "--group_id", str(args.group_id),
        "--img_size", str(args.img_size),
        "--sample_steps", str(args.sample_steps),
        "--guidance_scale", str(args.guidance_scale),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--base_channels", str(args.base_channels),
        "--num_timesteps", str(args.num_timesteps),
        "--n_res_blocks", str(args.n_res_blocks),
        "--seed", str(args.seed),
    ]
    if args.ckpt_root:
        cmd += ["--ckpt_root", args.ckpt_root]
    if args.ckpt_dir:
        cmd += ["--ckpt_dir", args.ckpt_dir]
    if args.samples_root:
        cmd += ["--samples_root", args.samples_root]
    return cmd


def build_xai_cmd(args):
    if args.xai_mode in ["ig", "ddim"]:
        script = "xai_eeg_importance_ig_ddim.py"
        cmd = [
            sys.executable,
            script,
            "--data_root", args.data_root,
            "--subject_id", str(args.subject_id),
            "--group_id", str(args.group_id),
            "--img_size", str(args.img_size),
            "--batch_size", str(args.xai_batch_size),
            "--num_workers", str(args.num_workers),
            "--base_channels", str(args.base_channels),
            "--num_timesteps", str(args.num_timesteps),
            "--n_res_blocks", str(args.n_res_blocks),
            "--seed", str(args.seed),
            "--guidance_scale", str(args.xai_guidance_scale),
            "--timestep", str(args.xai_timestep),
            "--loss", args.xai_loss,
            "--max_batches", str(args.xai_max_batches),
            "--mode", args.xai_mode,
            "--ig_steps", str(args.xai_ig_steps),
            "--ig_baseline", args.xai_ig_baseline,
            "--ddim_steps", str(args.xai_ddim_steps),
            "--out_dir", args.xai_out_dir,
        ]
    else:
        script = "xai_eeg_importance.py"
        cmd = [
            sys.executable,
            script,
            "--data_root", args.data_root,
            "--subject_id", str(args.subject_id),
            "--group_id", str(args.group_id),
            "--img_size", str(args.img_size),
            "--batch_size", str(args.xai_batch_size),
            "--num_workers", str(args.num_workers),
            "--base_channels", str(args.base_channels),
            "--num_timesteps", str(args.num_timesteps),
            "--n_res_blocks", str(args.n_res_blocks),
            "--seed", str(args.seed),
            "--guidance_scale", str(args.xai_guidance_scale),
            "--timestep", str(args.xai_timestep),
            "--loss", args.xai_loss,
            "--max_batches", str(args.xai_max_batches),
            "--out_dir", args.xai_out_dir,
        ]

    if args.ckpt_root:
        cmd += ["--ckpt_root", args.ckpt_root]
    if args.ckpt_dir:
        cmd += ["--ckpt_dir", args.ckpt_dir]
    if args.no_ema:
        cmd += ["--no_ema"]
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Run EEG-only sampling and XAI in one command."
    )
    parser.add_argument("--data_root", type=str, default="./preproc_data")
    parser.add_argument("--subject_id", type=int, required=True)
    parser.add_argument("--group_id", type=int, required=True)
    parser.add_argument("--img_size", type=int, default=128)

    parser.add_argument("--ckpt_root", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--samples_root", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--num_timesteps", type=int, default=2000)
    parser.add_argument("--n_res_blocks", type=int, default=2)
    parser.add_argument("--sample_steps", type=int, default=200)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--run_xai", action="store_true", default=False)
    parser.add_argument("--xai_mode", type=str, choices=["basic", "ig", "ddim"], default="basic")
    parser.add_argument("--xai_out_dir", type=str, default="./xai_eeg_importance")
    parser.add_argument("--xai_timestep", type=int, default=100)
    parser.add_argument("--xai_loss", type=str, choices=["l1", "l2"], default="l1")
    parser.add_argument("--xai_max_batches", type=int, default=5)
    parser.add_argument("--xai_batch_size", type=int, default=8)
    parser.add_argument("--xai_guidance_scale", type=float, default=None)
    parser.add_argument("--xai_ig_steps", type=int, default=20)
    parser.add_argument("--xai_ig_baseline", type=str, choices=["zero", "mean"], default="zero")
    parser.add_argument("--xai_ddim_steps", type=int, default=10)
    parser.add_argument("--no_ema", action="store_true", default=False)

    args = parser.parse_args()

    if args.xai_guidance_scale is None:
        args.xai_guidance_scale = args.guidance_scale

    sample_cmd = build_sample_cmd(args)
    print("Running sampling:", " ".join(sample_cmd))
    subprocess.run(sample_cmd, check=True)

    if args.run_xai:
        xai_cmd = build_xai_cmd(args)
        print("Running XAI:", " ".join(xai_cmd))
        subprocess.run(xai_cmd, check=True)


if __name__ == "__main__":
    main()
