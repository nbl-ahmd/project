#!/usr/bin/env python3
"""
Prepare a YOLO layout-detection dataset from corrected region annotations.

Input:
- page_manifest.csv from preprocess_pages.py or run_end_to_end.py
- region_manifest.csv with page-level boxes for handwritten regions

Output:
dataset_root/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
  data.yaml
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create YOLO dataset from region manifest.")
    parser.add_argument("--page-manifest", type=Path, required=True)
    parser.add_argument("--region-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name", type=str, default="handwritten_region")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_box(row: pd.Series, width: int, height: int) -> tuple[float, float, float, float]:
    x1 = max(0.0, min(float(row["x1_page"]), width - 1))
    y1 = max(0.0, min(float(row["y1_page"]), height - 1))
    x2 = max(1.0, min(float(row["x2_page"]), width))
    y2 = max(1.0, min(float(row["y2_page"]), height))
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

    pages = pd.read_csv(args.page_manifest)
    regions = pd.read_csv(args.region_manifest)

    required_pages = {"page_id", "width", "height"}
    missing_pages = required_pages - set(pages.columns)
    if missing_pages:
        raise ValueError(f"page manifest missing columns: {sorted(missing_pages)}")

    required_regions = {"page_id", "x1_page", "y1_page", "x2_page", "y2_page"}
    missing_regions = required_regions - set(regions.columns)
    if missing_regions:
        raise ValueError(f"region manifest missing columns: {sorted(missing_regions)}")

    page_path_col = "processed_path" if "processed_path" in pages.columns else "page_image_path"
    if page_path_col not in pages.columns:
        raise ValueError("page manifest must include processed_path or page_image_path")

    page_rows = pages.drop_duplicates("page_id").set_index("page_id")
    page_ids = sorted(set(regions["page_id"].astype(str)) & set(page_rows.index.astype(str)))
    if not page_ids:
        raise ValueError("No matching page_id values between page and region manifests.")
    rng.shuffle(page_ids)

    n_val = max(1, int(round(len(page_ids) * args.val_ratio))) if len(page_ids) > 1 else 0
    val_ids = set(page_ids[:n_val])

    for split in ["train", "val"]:
        (args.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    written = {"train": 0, "val": 0}
    for page_id in page_ids:
        page = page_rows.loc[page_id]
        src = Path(str(page[page_path_col]))
        if not src.exists():
            continue
        width = int(page["width"])
        height = int(page["height"])
        split = "val" if page_id in val_ids else "train"

        image_name = f"{page_id}{src.suffix.lower() if src.suffix else '.jpg'}"
        dst_img = args.output_dir / "images" / split / image_name
        shutil.copy2(src, dst_img)

        label_lines = []
        for _, region in regions[regions["page_id"].astype(str) == str(page_id)].iterrows():
            try:
                cx, cy, bw, bh = normalize_box(region, width, height)
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
    print("YOLO layout dataset prepared.")
    print(f"Train pages: {written['train']}")
    print(f"Val pages: {written['val']}")
    print(f"Data YAML: {args.output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
