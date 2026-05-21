#!/usr/bin/env python3
"""
Create a train-only augmented copy of a word-level OCR dataset.

Expected dataset layout:
output_root/
  Training/training_words/*.png
  Training/training_labels.csv
  Validation/validation_words/*.png
  Validation/validation_labels.csv
  Testing/testing_words/*.png
  Testing/testing_labels.csv

Validation and testing are copied unchanged so evaluation stays honest.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


SPLITS = {
    "Training": ("training_words", "training_labels.csv"),
    "Validation": ("validation_words", "validation_labels.csv"),
    "Testing": ("testing_words", "testing_labels.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment word OCR training split only.")
    parser.add_argument("--input-root", type=Path, required=True, help="Source OCR dataset root")
    parser.add_argument("--output-root", type=Path, required=True, help="Augmented dataset output root")
    parser.add_argument("--augmentations-per-image", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rotate", type=float, default=4.0, help="Max absolute rotation in degrees")
    parser.add_argument("--max-shift-ratio", type=float, default=0.06, help="Max x/y shift as image-size ratio")
    parser.add_argument("--scale-min", type=float, default=0.94)
    parser.add_argument("--scale-max", type=float, default=1.08)
    parser.add_argument("--brightness", type=float, default=0.18, help="Brightness jitter fraction")
    parser.add_argument("--contrast", type=float, default=0.18, help="Contrast jitter fraction")
    parser.add_argument("--blur-prob", type=float, default=0.15)
    parser.add_argument("--noise-prob", type=float, default=0.15)
    return parser.parse_args()


def copy_split(input_root: Path, output_root: Path, split: str) -> pd.DataFrame:
    image_dir_name, csv_name = SPLITS[split]
    src_split = input_root / split
    dst_split = output_root / split
    src_images = src_split / image_dir_name
    dst_images = dst_split / image_dir_name
    src_csv = src_split / csv_name

    if not src_csv.exists():
        raise FileNotFoundError(src_csv)
    if not src_images.exists():
        raise FileNotFoundError(src_images)

    dst_images.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src_csv)
    if "IMAGE" not in df.columns or "MEDICINE_NAME" not in df.columns:
        raise ValueError(f"{src_csv} must contain IMAGE and MEDICINE_NAME columns")

    rows = []
    for _, row in df.iterrows():
        src = src_images / str(row["IMAGE"])
        if not src.exists():
            continue
        dst = dst_images / src.name
        shutil.copy2(src, dst)
        rows.append({"IMAGE": dst.name, "MEDICINE_NAME": row["MEDICINE_NAME"]})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(dst_split / csv_name, index=False)
    return out_df


def affine_word_image(image: np.ndarray, rng: np.random.Generator, args: argparse.Namespace) -> np.ndarray:
    h, w = image.shape[:2]
    angle = float(rng.uniform(-args.max_rotate, args.max_rotate))
    scale = float(rng.uniform(args.scale_min, args.scale_max))
    tx = float(rng.uniform(-args.max_shift_ratio, args.max_shift_ratio) * w)
    ty = float(rng.uniform(-args.max_shift_ratio, args.max_shift_ratio) * h)

    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    warped = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    alpha = float(rng.uniform(1.0 - args.contrast, 1.0 + args.contrast))
    beta = float(rng.uniform(-255 * args.brightness, 255 * args.brightness))
    warped = cv2.convertScaleAbs(warped, alpha=alpha, beta=beta)

    if rng.random() < args.blur_prob:
        warped = cv2.GaussianBlur(warped, (3, 3), 0)

    if rng.random() < args.noise_prob:
        noise = rng.normal(0, 5, warped.shape).astype(np.int16)
        warped = np.clip(warped.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return warped


def augment_training_split(input_root: Path, output_root: Path, args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    image_dir_name, csv_name = SPLITS["Training"]
    src_images = input_root / "Training" / image_dir_name
    dst_images = output_root / "Training" / image_dir_name
    labels_path = output_root / "Training" / csv_name
    df = pd.read_csv(labels_path)

    augmented_rows = []
    for _, row in df.iterrows():
        image_name = str(row["IMAGE"])
        src = src_images / image_name
        image = cv2.imread(str(src))
        if image is None:
            continue
        stem = Path(image_name).stem
        for idx in range(args.augmentations_per_image):
            aug_name = f"{stem}_aug{idx:02d}.png"
            aug = affine_word_image(image, rng, args)
            cv2.imwrite(str(dst_images / aug_name), aug)
            augmented_rows.append({"IMAGE": aug_name, "MEDICINE_NAME": row["MEDICINE_NAME"]})

    if augmented_rows:
        out = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
        out.to_csv(labels_path, index=False)
    return len(augmented_rows)


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    train_df = copy_split(args.input_root, args.output_root, "Training")
    val_df = copy_split(args.input_root, args.output_root, "Validation")
    test_df = copy_split(args.input_root, args.output_root, "Testing")
    added = augment_training_split(args.input_root, args.output_root, args)

    print("OCR dataset augmentation complete.")
    print(f"Original train: {len(train_df)}")
    print(f"Augmented train rows added: {added}")
    print(f"Final train: {len(train_df) + added}")
    print(f"Validation copied unchanged: {len(val_df)}")
    print(f"Testing copied unchanged: {len(test_df)}")
    print(f"Output root: {args.output_root}")


if __name__ == "__main__":
    main()
