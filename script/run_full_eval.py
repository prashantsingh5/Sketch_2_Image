"""Run evaluation and update README benchmark row in one command.

Directory mode:
python script/run_full_eval.py \
  --real-dir dataset_for_pix2pix/B/val \
  --generated-dir pytorch-CycleGAN-and-pix2pix/output_images

Single-image mode:
python script/run_full_eval.py \
  --real-image assets/sample/before.jpg \
  --generated-image assets/sample/after.jpg
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate outputs and update README benchmark")
    parser.add_argument("--real-dir", type=Path, default=None)
    parser.add_argument("--generated-dir", type=Path, default=None)
    parser.add_argument("--real-image", type=Path, default=None)
    parser.add_argument("--generated-image", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=Path("metrics/latest_metrics.json"))
    parser.add_argument("--experiment", type=str, default="Baseline Pix2Pix (default config)")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print("[run]", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()

    evaluate_cmd = [sys.executable, "script/evaluate_metrics.py"]
    if args.real_dir and args.generated_dir:
        evaluate_cmd += ["--real-dir", str(args.real_dir), "--generated-dir", str(args.generated_dir)]
    elif args.real_image and args.generated_image:
        evaluate_cmd += ["--real-image", str(args.real_image), "--generated-image", str(args.generated_image)]
    else:
        raise ValueError(
            "Provide either --real-dir/--generated-dir or --real-image/--generated-image"
        )

    evaluate_cmd += ["--output-json", str(args.output_json), "--max-images", str(args.max_images)]
    if args.device:
        evaluate_cmd += ["--device", args.device]

    update_cmd = [
        sys.executable,
        "script/update_benchmark_table.py",
        "--metrics-json",
        str(args.output_json),
        "--experiment",
        args.experiment,
        "--split",
        args.split,
    ]

    _run(evaluate_cmd)
    _run(update_cmd)
    print("[done] Metrics computed and README benchmark row updated.")


if __name__ == "__main__":
    main()
