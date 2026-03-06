"""Prepare sketch-to-image datasets for Pix2Pix training.

This script creates three artifacts from an input image directory:
1) `dataset_sketch/` with generated pencil-sketch images.
2) `dataset_for_pix2pix/` with `A/` (sketch) and `B/` (real) split folders.
3) `dataset_ab/` with concatenated AB images expected by Pix2Pix.
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in VALID_EXTENSIONS


def _collect_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and _is_image(p)])


def generate_sketches(input_folder: Path, output_folder: Path) -> int:
    output_folder.mkdir(parents=True, exist_ok=True)
    generated = 0

    for image_path in _collect_images(input_folder):
        relative_parent = image_path.parent.relative_to(input_folder)
        sketch_subfolder = output_folder / relative_parent
        sketch_subfolder.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[warn] Failed to load image: {image_path}")
            continue

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inverted_image = 255 - gray_image
        blurred_image = cv2.GaussianBlur(inverted_image, (21, 21), 0)
        inverted_blurred = 255 - blurred_image
        sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)

        sketch_name = f"sketch_{image_path.stem}.jpg"
        sketch_path = sketch_subfolder / sketch_name
        cv2.imwrite(str(sketch_path), sketch)
        generated += 1

    return generated


def _paired_records(real_folder: Path, sketch_folder: Path) -> List[Tuple[Path, Path]]:
    real_index: Dict[str, Path] = {}
    for real in _collect_images(real_folder):
        real_index[real.stem.lower()] = real

    pairs: List[Tuple[Path, Path]] = []
    for sketch in _collect_images(sketch_folder):
        if not sketch.stem.lower().startswith("sketch_"):
            continue
        source_name = sketch.stem[7:].lower()
        real = real_index.get(source_name)
        if real is not None:
            pairs.append((real, sketch))

    return sorted(pairs, key=lambda x: x[0].name.lower())


def _split_pairs(
    pairs: List[Tuple[Path, Path]], val_ratio: float, test_ratio: float, seed: int
) -> Dict[str, List[Tuple[Path, Path]]]:
    if not pairs:
        raise ValueError("No matched image pairs found. Check your dataset structure.")
    if (val_ratio + test_ratio) >= 1:
        raise ValueError("val_ratio + test_ratio must be < 1.")

    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)

    total = len(shuffled)
    n_test = int(total * test_ratio)
    n_val = int(total * val_ratio)
    n_train = total - n_val - n_test

    if n_train <= 0:
        raise ValueError("Training split is empty. Lower val/test ratios or add more data.")

    train_pairs = shuffled[:n_train]
    val_pairs = shuffled[n_train : n_train + n_val]
    test_pairs = shuffled[n_train + n_val :]

    return {"train": train_pairs, "val": val_pairs, "test": test_pairs}


def prepare_pix2pix_dataset(
    real_folder: Path,
    sketch_folder: Path,
    output_folder: Path,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, int]:
    a_folder = output_folder / "A"
    b_folder = output_folder / "B"

    for split in ["train", "val", "test"]:
        (a_folder / split).mkdir(parents=True, exist_ok=True)
        (b_folder / split).mkdir(parents=True, exist_ok=True)

    pairs = _paired_records(real_folder, sketch_folder)
    print(f"[info] Matched real/sketch pairs: {len(pairs)}")
    split_records = _split_pairs(pairs, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for split, records in split_records.items():
        for idx, (real, sketch) in enumerate(records, start=1):
            filename = f"{idx:06d}.jpg"
            shutil.copy2(real, b_folder / split / filename)
            shutil.copy2(sketch, a_folder / split / filename)
            counts[split] += 1

    return counts


def combine_a_and_b(folder_a: Path, folder_b: Path, output_folder: Path) -> Dict[str, int]:
    output_folder.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for split in ["train", "val", "test"]:
        split_a = folder_a / split
        split_b = folder_b / split
        split_ab = output_folder / split
        split_ab.mkdir(parents=True, exist_ok=True)

        a_images = sorted([p for p in split_a.glob("*.jpg") if p.is_file()])
        b_images = sorted([p for p in split_b.glob("*.jpg") if p.is_file()])

        if len(a_images) != len(b_images):
            raise ValueError(f"Split '{split}' is misaligned: {len(a_images)} A vs {len(b_images)} B")

        for a_path, b_path in zip(a_images, b_images):
            im_a = cv2.imread(str(a_path))
            im_b = cv2.imread(str(b_path))
            if im_a is None or im_b is None:
                print(f"[warn] Skipping unreadable pair: {a_path.name}")
                continue

            if im_a.shape[0] != im_b.shape[0]:
                # Keep width ratio while aligning heights for safe concatenation.
                target_h = min(im_a.shape[0], im_b.shape[0])
                im_a = cv2.resize(im_a, (int(im_a.shape[1] * target_h / im_a.shape[0]), target_h))
                im_b = cv2.resize(im_b, (int(im_b.shape[1] * target_h / im_b.shape[0]), target_h))

            combined = np.concatenate([im_a, im_b], axis=1)
            cv2.imwrite(str(split_ab / a_path.name), combined)
            counts[split] += 1

    return counts


def move_dataset_to_pytorch_folder(dataset_folder: Path, pytorch_folder: Path, dataset_name: str) -> Path:
    datasets_folder = pytorch_folder / "datasets"
    datasets_folder.mkdir(parents=True, exist_ok=True)

    destination = datasets_folder / dataset_name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(dataset_folder, destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Pix2Pix dataset from real images")
    parser.add_argument("--base-folder", type=Path, default=Path.cwd(), help="Project base directory")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"), help="Input dataset folder (relative to base-folder if not absolute)")
    parser.add_argument("--sketch-dir", type=Path, default=Path("dataset_sketch"), help="Generated sketch folder")
    parser.add_argument("--pix2pix-dir", type=Path, default=Path("dataset_for_pix2pix"), help="A/B structured output folder")
    parser.add_argument("--ab-dir", type=Path, default=Path("dataset_ab"), help="Combined AB output folder")
    parser.add_argument("--pix2pix-repo", type=Path, default=Path("pytorch-CycleGAN-and-pix2pix"), help="Path to pytorch-CycleGAN-and-pix2pix repo")
    parser.add_argument("--dataset-name", default="sketch2image", help="Destination dataset name under pix2pix repo datasets/")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.0, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits")
    return parser.parse_args()


def _resolve(base: Path, value: Path) -> Path:
    return value if value.is_absolute() else (base / value)


def main() -> None:
    args = parse_args()
    base_folder = args.base_folder.resolve()

    real_images_folder = _resolve(base_folder, args.dataset_dir)
    sketch_images_folder = _resolve(base_folder, args.sketch_dir)
    pix2pix_dataset_folder = _resolve(base_folder, args.pix2pix_dir)
    combined_dataset_folder = _resolve(base_folder, args.ab_dir)
    pix2pix_repo_folder = _resolve(base_folder, args.pix2pix_repo)

    if not real_images_folder.exists():
        raise FileNotFoundError(f"Input dataset folder not found: {real_images_folder}")
    if not pix2pix_repo_folder.exists():
        raise FileNotFoundError(f"Pix2Pix repo folder not found: {pix2pix_repo_folder}")

    print(f"[step] Generating sketches from: {real_images_folder}")
    generated = generate_sketches(real_images_folder, sketch_images_folder)
    print(f"[done] Generated sketches: {generated}")

    print("[step] Building A/B dataset splits")
    counts = prepare_pix2pix_dataset(
        real_images_folder,
        sketch_images_folder,
        pix2pix_dataset_folder,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"[done] Split counts: train={counts['train']}, val={counts['val']}, test={counts['test']}")

    print("[step] Combining A and B into AB format")
    ab_counts = combine_a_and_b(
        pix2pix_dataset_folder / "A",
        pix2pix_dataset_folder / "B",
        combined_dataset_folder,
    )
    print(f"[done] AB counts: train={ab_counts['train']}, val={ab_counts['val']}, test={ab_counts['test']}")

    print("[step] Copying AB dataset into Pix2Pix datasets folder")
    destination = move_dataset_to_pytorch_folder(combined_dataset_folder, pix2pix_repo_folder, args.dataset_name)
    print(f"[done] Dataset available at: {destination}")


if __name__ == "__main__":
    main()
