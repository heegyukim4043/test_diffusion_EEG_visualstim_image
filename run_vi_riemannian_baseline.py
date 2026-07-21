"""Run VI Riemannian baseline across subjects/configs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1,2,9,18,24,28,29")
    parser.add_argument("--vi_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--features", default="cov,filterbank")
    parser.add_argument("--classifiers", default="logreg,linsvm,lda")
    parser.add_argument("--shrinkages", default="0.05,0.10,0.20")
    parser.add_argument("--n_permutations", type=int, default=0)
    parser.add_argument("--script", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_csv(value: str, cast=str):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    subjects = parse_csv(args.subjects, int)
    features = parse_csv(args.features, str)
    classifiers = parse_csv(args.classifiers, str)
    shrinkages = parse_csv(args.shrinkages, float)
    script = Path(args.script) if args.script else Path(__file__).with_name("train_vi_riemannian_baseline.py")
    if not script.is_file():
        raise FileNotFoundError(script)

    for sid in subjects:
        for feature in features:
            for classifier in classifiers:
                for shrinkage in shrinkages:
                    out_dir = (
                        Path(args.out_root)
                        / f"seed{args.seed}"
                        / f"S{sid:02d}"
                        / feature
                        / classifier
                        / f"shrink{shrinkage:g}"
                    )
                    metrics = out_dir / "metrics.json"
                    if metrics.exists() and not args.overwrite:
                        print(f"[SKIP] S{sid:02d} {feature} {classifier} shrink={shrinkage:g}", flush=True)
                        continue
                    print("=" * 90, flush=True)
                    print(f"START S{sid:02d} {feature} {classifier} shrink={shrinkage:g}", flush=True)
                    print("=" * 90, flush=True)
                    cmd = [
                        sys.executable,
                        "-u",
                        str(script),
                        "--subject_id",
                        str(sid),
                        "--vi_root",
                        args.vi_root,
                        "--out_dir",
                        str(out_dir),
                        "--seed",
                        str(args.seed),
                        "--feature",
                        feature,
                        "--classifier",
                        classifier,
                        "--shrinkage",
                        str(shrinkage),
                        "--ch_zscore",
                    ]
                    if args.n_permutations > 0:
                        cmd += ["--n_permutations", str(args.n_permutations)]
                    if args.overwrite:
                        cmd.append("--overwrite")
                    result = subprocess.run(cmd)
                    print("RETURN CODE:", result.returncode, flush=True)
                    if result.returncode != 0:
                        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
