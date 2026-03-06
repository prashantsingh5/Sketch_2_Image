"""Compute evaluation metrics for image-to-image translation outputs.

Metrics:
- FID (distribution-level, lower is better)
- LPIPS (perceptual distance, lower is better)
- SSIM (structural similarity, higher is better)

Example:
python script/evaluate_metrics.py \
  --real-dir pytorch-CycleGAN-and-pix2pix/datasets/sketch2image/val \
  --generated-dir pytorch-CycleGAN-and-pix2pix/results/sketch2image/test_latest/images \
  --output-json metrics/latest_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError as exc:  # pragma: no cover - import safety
    raise ImportError("Missing dependency 'scikit-image'. Install with: pip install -r requirements.txt") from exc

try:
    import torch
except ImportError as exc:  # pragma: no cover - import safety
    raise ImportError("Missing dependency 'torch'. Install with: pip install -r requirements.txt") from exc

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError as exc:  # pragma: no cover - import safety
    raise ImportError("Missing dependency 'torchmetrics' and 'torch-fidelity'. Install with: pip install -r requirements.txt") from exc

try:
    import lpips
except ImportError as exc:  # pragma: no cover - import safety
    raise ImportError("Missing dependency 'lpips'. Install with: pip install -r requirements.txt") from exc

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated images with FID/LPIPS/SSIM")
    parser.add_argument("--real-dir", type=Path, default=None, help="Directory with ground-truth real images")
    parser.add_argument("--generated-dir", type=Path, default=None, help="Directory with generated images")
    parser.add_argument("--real-image", type=Path, default=None, help="Single ground-truth image path")
    parser.add_argument("--generated-image", type=Path, default=None, help="Single generated image path")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to write JSON metrics")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of image pairs (0 = all)")
    return parser.parse_args()


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in VALID_EXT


def _normalize_stem(stem: str) -> str:
    lowered = stem.lower()
    for suffix in ["_fake_b", "_real_b", "_real_a", "_fake_a", "_input", "_output"]:
        if lowered.endswith(suffix):
            return lowered[: -len(suffix)]
    return lowered


def _index_by_stem(folder: Path) -> Dict[str, Path]:
    files = [p for p in folder.rglob("*") if p.is_file() and _is_image(p)]
    index: Dict[str, Path] = {}
    for path in files:
        key = _normalize_stem(path.stem)
        # Keep first occurrence to avoid accidental overwrite from duplicates.
        index.setdefault(key, path)
    return index


def _pair_images(real_dir: Path, generated_dir: Path, max_images: int) -> List[Tuple[Path, Path]]:
    real_index = _index_by_stem(real_dir)
    gen_index = _index_by_stem(generated_dir)

    common = sorted(set(real_index.keys()) & set(gen_index.keys()))
    if not common:
        raise ValueError("No matching image pairs found by filename stem.")

    if max_images > 0:
        common = common[:max_images]

    return [(real_index[k], gen_index[k]) for k in common]


def _pair_single_images(real_image: Path, generated_image: Path) -> List[Tuple[Path, Path]]:
    if not real_image.exists():
        raise FileNotFoundError(f"real-image not found: {real_image}")
    if not generated_image.exists():
        raise FileNotFoundError(f"generated-image not found: {generated_image}")
    if not _is_image(real_image) or not _is_image(generated_image):
        raise ValueError("Both --real-image and --generated-image must be valid image files.")
    return [(real_image, generated_image)]


def _to_tensor_for_fid(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1)


def _to_tensor_for_lpips(img: Image.Image, device: str) -> torch.Tensor:
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return (tensor * 2.0 - 1.0).to(device)


def _load_pair(real_path: Path, gen_path: Path) -> Tuple[Image.Image, Image.Image]:
    real_img = Image.open(real_path).convert("RGB")
    gen_img = Image.open(gen_path).convert("RGB")
    if real_img.size != gen_img.size:
        gen_img = gen_img.resize(real_img.size, Image.BICUBIC)
    return real_img, gen_img


def compute_metrics(pairs: List[Tuple[Path, Path]], device: str) -> Dict[str, float]:
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    lpips_metric = lpips.LPIPS(net="alex").to(device)

    lpips_scores: List[float] = []
    ssim_scores: List[float] = []

    with torch.no_grad():
        for real_path, gen_path in pairs:
            real_img, gen_img = _load_pair(real_path, gen_path)

            real_fid = _to_tensor_for_fid(real_img).unsqueeze(0).to(device)
            gen_fid = _to_tensor_for_fid(gen_img).unsqueeze(0).to(device)
            fid_metric.update(real_fid, real=True)
            fid_metric.update(gen_fid, real=False)

            real_lp = _to_tensor_for_lpips(real_img, device)
            gen_lp = _to_tensor_for_lpips(gen_img, device)
            lpips_scores.append(float(lpips_metric(real_lp, gen_lp).item()))

            real_np = np.array(real_img.convert("L"), dtype=np.float32)
            gen_np = np.array(gen_img.convert("L"), dtype=np.float32)
            score = ssim(real_np, gen_np, data_range=255.0)
            ssim_scores.append(float(score))

    fid_value = float(fid_metric.compute().item())
    lpips_value = float(np.mean(lpips_scores))
    ssim_value = float(np.mean(ssim_scores))

    return {
        "num_pairs": len(pairs),
        "fid": fid_value,
        "lpips": lpips_value,
        "ssim": ssim_value,
    }


def main() -> None:
    args = parse_args()

    dir_mode = args.real_dir is not None and args.generated_dir is not None
    file_mode = args.real_image is not None and args.generated_image is not None

    if not (dir_mode or file_mode):
        raise ValueError(
            "Provide either directory mode (--real-dir and --generated-dir) "
            "or single-image mode (--real-image and --generated-image)."
        )

    if dir_mode:
        real_dir = args.real_dir.resolve()
        generated_dir = args.generated_dir.resolve()
        if not real_dir.exists():
            raise FileNotFoundError(f"real-dir not found: {real_dir}")
        if not generated_dir.exists():
            raise FileNotFoundError(f"generated-dir not found: {generated_dir}")
        pairs = _pair_images(real_dir, generated_dir, args.max_images)
        real_source = str(real_dir)
        generated_source = str(generated_dir)
    else:
        real_image = args.real_image.resolve()
        generated_image = args.generated_image.resolve()
        pairs = _pair_single_images(real_image, generated_image)
        real_source = str(real_image)
        generated_source = str(generated_image)

    metrics = compute_metrics(pairs, args.device)

    print("Evaluation Results")
    print(f"Pairs : {metrics['num_pairs']}")
    print(f"FID   : {metrics['fid']:.4f} (lower is better)")
    print(f"LPIPS : {metrics['lpips']:.4f} (lower is better)")
    print(f"SSIM  : {metrics['ssim']:.4f} (higher is better)")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_pairs": metrics["num_pairs"],
            "fid": metrics["fid"],
            "lpips": metrics["lpips"],
            "ssim": metrics["ssim"],
            "real_source": real_source,
            "generated_source": generated_source,
        }
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved metrics JSON to: {args.output_json}")


if __name__ == "__main__":
    main()
