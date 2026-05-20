#!/usr/bin/env python3
"""
Prepare a YOLO line-detection dataset from corrected line annotations.

Input:
- line_manifest.csv from manual line annotation or segment_lines.py

Output:
dataset_root/
  images/train/*.png
  images/val/*.png
  labels/train/*.txt
  labels/val/*.txt
  data.yaml
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create YOLO dataset from line manifest.")
    parser.add_argument("--line-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name", type=str, default="handwritten_line")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_box(row: pd.Series, width: int, height: int) -> tuple[float, float, float, float]:
    x1 = max(0.0, min(float(row["x1_region"]), width - 1))
    y1 = max(0.0, min(float(row["y1_region"]), height - 1))
    x2 = max(1.0, min(float(row["x2_region"]), width))
    y2 = max(1.0, min(float(row["y2_region"]), height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid box")
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return cx, cy, bw, bh


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    lines = pd.read_csv(args.line_manifest)
    required = {"region_id", "region_image_path", "x1_region", "y1_region", "x2_region", "y2_region"}
    missing = required - set(lines.columns)
    if missing:
        raise ValueError(f"line manifest missing columns: {sorted(missing)}")

    region_ids = sorted(lines["region_id"].astype(str).unique())
    if not region_ids:
        raise ValueError("No region_id values found in line manifest.")
    rng.shuffle(region_ids)

    n_val = max(1, int(round(len(region_ids) * args.val_ratio))) if len(region_ids) > 1 else 0
    val_ids = set(region_ids[:n_val])

    for split in ["train", "val"]:
        (args.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    written = {"train": 0, "val": 0}
    for region_id in region_ids:
        region_rows = lines[lines["region_id"].astype(str) == region_id]
        src = Path(str(region_rows.iloc[0]["region_image_path"]))
        if not src.exists():
            continue
        image = cv2.imread(str(src))
        if image is None:
            continue
        height, width = image.shape[:2]
        split = "val" if region_id in val_ids else "train"

        image_name = f"{region_id}{src.suffix.lower() if src.suffix else '.png'}"
        dst_img = args.output_dir / "images" / split / image_name
        shutil.copy2(src, dst_img)

        label_lines = []
        for _, line in region_rows.iterrows():
            try:
                cx, cy, bw, bh = normalize_box(line, width, height)
            except ValueError:
                continue
            label_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not label_lines:
            dst_img.unlink(missing_ok=True)
            continue
        (args.output_dir / "labels" / split / f"{Path(image_name).stem}.txt").write_text(
            "\n".join(label_lines) + "\n",
            encoding="utf-8",
        )
        written[split] += 1

    yaml_text = f"""path: {args.output_dir.resolve()}
train: images/train
val: images/val
names:
  0: {args.class_name}
"""
    (args.output_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print("YOLO line dataset prepared.")
    print(f"Train regions: {written['train']}")
    print(f"Val regions: {written['val']}")
    print(f"Data YAML: {args.output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
