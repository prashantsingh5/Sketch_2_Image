"""Update README benchmark table from a metrics JSON file.

Example:
python script/update_benchmark_table.py \
  --metrics-json metrics/latest_metrics.json \
  --readme README.md \
  --experiment "Baseline Pix2Pix (default config)" \
  --split val
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update README benchmark table from metrics JSON")
    parser.add_argument("--metrics-json", type=Path, required=True, help="Path to metrics JSON")
    parser.add_argument("--readme", type=Path, default=Path("README.md"), help="Path to README file")
    parser.add_argument(
        "--experiment",
        type=str,
        default="Baseline Pix2Pix (default config)",
        help="Experiment name row key in benchmark table",
    )
    parser.add_argument("--split", type=str, default="val", help="Dataset split label for benchmark table")
    return parser.parse_args()


def _format_row(experiment: str, split: str, fid: float, lpips: float, ssim: float, notes: str) -> str:
    return f"| {experiment} | {split} | {fid:.4f} | {lpips:.4f} | {ssim:.4f} | {notes} |"


def main() -> None:
    args = parse_args()
    metrics_path = args.metrics_json.resolve()
    readme_path = args.readme.resolve()

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics JSON not found: {metrics_path}")
    if not readme_path.exists():
        raise FileNotFoundError(f"README not found: {readme_path}")

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    fid = float(payload["fid"])
    lpips = float(payload["lpips"])
    ssim = float(payload["ssim"])
    num_pairs = int(payload.get("num_pairs", 0))
    notes = f"Auto-updated from metrics JSON ({num_pairs} pairs)"

    new_row = _format_row(args.experiment, args.split, fid, lpips, ssim, notes)

    readme_text = readme_path.read_text(encoding="utf-8")
    pattern = re.compile(r"^\|\s*" + re.escape(args.experiment) + r"\s*\|.*$", flags=re.MULTILINE)

    if pattern.search(readme_text):
        updated = pattern.sub(new_row, readme_text)
    else:
        marker = "|---|---|---:|---:|---:|---|"
        if marker not in readme_text:
            raise ValueError("Benchmark table marker not found in README.")
        updated = readme_text.replace(marker, marker + "\n" + new_row)

    readme_path.write_text(updated, encoding="utf-8")
    print(f"Updated benchmark row in: {readme_path}")


if __name__ == "__main__":
    main()
