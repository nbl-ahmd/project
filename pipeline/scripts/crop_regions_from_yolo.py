#!/usr/bin/env python3
"""
Crop handwritten regions from page images using YOLO-format layout labels.

Expected labels:
- one .txt per image
- row format: class_id cx cy w h (normalized)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import pandas as pd
from tqdm import tqdm


VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> Iterable[Path]:
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            yield p


def read_class_map(path: Path) -> list[str]:
    classes = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not classes:
        raise ValueError(f"No classes found in {path}")
    return classes


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x1 = int((cx - w / 2.0) * img_w)
    y1 = int((cy - h / 2.0) * img_h)
    x2 = int((cx + w / 2.0) * img_w)
    y2 = int((cy + h / 2.0) * img_h)
    return x1, y1, x2, y2


def clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(1, min(x2, w))
    y2 = max(1, min(y2, h))
    return x1, y1, x2, y2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop target regions from YOLO layout labels.")
    parser.add_argument("--pages-dir", type=Path, required=True, help="Processed page images directory.")
    parser.add_argument("--labels-dir", type=Path, required=True, help="YOLO txt labels directory.")
    parser.add_argument("--class-map", type=Path, required=True, help="Class names file.")
    parser.add_argument(
        "--target-label",
        type=str,
        default="handwritten_region",
        help="Class label to crop.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output region crop directory.")
    parser.add_argument("--manifest-out", type=Path, required=True, help="Region manifest CSV.")
    parser.add_argument("--padding-px", type=int, default=8, help="Padding around detected region.")
    parser.add_argument("--min-area", type=int, default=2000, help="Minimum accepted crop area in pixels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)

    classes = read_class_map(args.class_map)
    if args.target_label not in classes:
        raise ValueError(f"target_label='{args.target_label}' not found in class map: {classes}")
    target_class_id = classes.index(args.target_label)

    rows = []
    image_paths = list(list_images(args.pages_dir))
    if not image_paths:
        raise FileNotFoundError(f"No page images found in {args.pages_dir}")

    for img_path in tqdm(image_paths, desc="Cropping regions"):
        page = cv2.imread(str(img_path))
        if page is None:
            continue
        h, w = page.shape[:2]

        label_path = args.labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        lines = [x.strip() for x in label_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        region_idx = 0
        for row in lines:
            parts = row.split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            if cls_id != target_class_id:
                continue

            cx, cy, bw, bh = map(float, parts[1:5])
            x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, w, h)
            x1 -= args.padding_px
            y1 -= args.padding_px
            x2 += args.padding_px
            y2 += args.padding_px
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)

            if (x2 - x1) * (y2 - y1) < args.min_area:
                continue

            crop = page[y1:y2, x1:x2]
            page_id = img_path.stem
            region_id = f"{page_id}_r{region_idx:02d}"
            crop_name = f"{region_id}.png"
            crop_path = args.output_dir / crop_name
            cv2.imwrite(str(crop_path), crop)

            rows.append(
                {
                    "region_id": region_id,
                    "page_id": page_id,
                    "region_label": args.target_label,
                    "page_image_path": str(img_path),
                    "region_image_path": str(crop_path),
                    "x1_page": x1,
                    "y1_page": y1,
                    "x2_page": x2,
                    "y2_page": y2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                }
            )
            region_idx += 1

    df = pd.DataFrame(rows).sort_values(["page_id", "region_id"]).reset_index(drop=True)
    df.to_csv(args.manifest_out, index=False)
    print(f"Cropped regions: {len(df)}")
    print(f"Manifest saved: {args.manifest_out}")


if __name__ == "__main__":
    main()

