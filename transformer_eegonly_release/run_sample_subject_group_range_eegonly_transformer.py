# run_sample_subject_group_range_eegonly_transformer.py
import argparse
import subprocess
import sys


def parse_range(value: str):
    if "," in value:
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    if "-" in value:
        a, b = value.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(value)]


def main():
    parser = argparse.ArgumentParser(
        description="Run EEG-only Transformer sampling over subject/group ranges."
    )
    parser.add_argument("--data_root", type=str, default="./preproc_data")
    parser.add_argument("--subjects", type=str, required=True,
                        help="e.g., 1-5 or 1,2,3")
    parser.add_argument("--groups", type=str, required=True,
                        help="e.g., 1-3 or 1,2,3")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--sample_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--num_timesteps", type=int, default=2000)
    parser.add_argument("--n_res_blocks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_root", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--samples_root", type=str, default=None)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--eeg_tf_heads", type=int, default=4)
    parser.add_argument("--eeg_tf_layers", type=int, default=2)
    parser.add_argument("--eeg_tf_dropout", type=float, default=0.1)

    args = parser.parse_args()

    subjects = parse_range(args.subjects)
    groups = parse_range(args.groups)

    for sid in subjects:
        for gid in groups:
            cmd = [
                sys.executable,
                "sample_subject_all_group_128_eegonly_transformer.py",
                "--data_root", args.data_root,
                "--subject_id", str(sid),
                "--group_id", str(gid),
                "--img_size", str(args.img_size),
                "--sample_steps", str(args.sample_steps),
                "--batch_size", str(args.batch_size),
                "--num_workers", str(args.num_workers),
                "--base_channels", str(args.base_channels),
                "--num_timesteps", str(args.num_timesteps),
                "--n_res_blocks", str(args.n_res_blocks),
                "--seed", str(args.seed),
                "--guidance_scale", str(args.guidance_scale),
                "--eeg_tf_heads", str(args.eeg_tf_heads),
                "--eeg_tf_layers", str(args.eeg_tf_layers),
                "--eeg_tf_dropout", str(args.eeg_tf_dropout),
            ]
            if args.ckpt_root:
                cmd += ["--ckpt_root", args.ckpt_root]
            if args.ckpt_dir:
                cmd += ["--ckpt_dir", args.ckpt_dir]
            if args.samples_root:
                cmd += ["--samples_root", args.samples_root]

            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
