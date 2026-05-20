#!/usr/bin/env python3
"""
Preprocess raw prescription page images:
- optional resize
- deskew
- denoise
- contrast enhancement

Outputs:
- processed page images
- manifest CSV
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> Iterable[Path]:
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            yield p


def estimate_skew_angle_deg(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(th > 0))
    if coords.shape[0] < 500:
        return 0.0

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return float(angle)


def rotate_keep_canvas(image_bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 0.1:
        return image_bgr
    h, w = image_bgr.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        image_bgr,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def enhance_contrast(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge([l, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess_page(image_bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = image_bgr.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    angle = estimate_skew_angle_deg(image_bgr)
    image_bgr = rotate_keep_canvas(image_bgr, angle)
    image_bgr = cv2.fastNlMeansDenoisingColored(image_bgr, None, 5, 5, 7, 21)
    image_bgr = enhance_contrast(image_bgr)
    return image_bgr, angle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw prescription pages.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder with raw page images.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder for processed page images.")
    parser.add_argument("--manifest-out", type=Path, required=True, help="CSV output path.")
    parser.add_argument("--max-side", type=int, default=2200, help="Max dimension for output pages.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="Output JPEG quality.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    image_paths = list(list_images(args.input_dir))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    for src_path in tqdm(image_paths, desc="Preprocessing pages"):
        image = cv2.imread(str(src_path))
        if image is None:
            continue

        processed, angle = preprocess_page(image, args.max_side)
        page_id = src_path.stem.replace(" ", "_")
        out_name = f"{page_id}.jpg"
        out_path = args.output_dir / out_name
        cv2.imwrite(str(out_path), processed, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])

        ph, pw = processed.shape[:2]
        rows.append(
            {
                "page_id": page_id,
                "source_path": str(src_path),
                "processed_path": str(out_path),
                "width": pw,
                "height": ph,
                "deskew_angle_deg": round(angle, 3),
            }
        )

    df = pd.DataFrame(rows).sort_values("page_id").reset_index(drop=True)
    df.to_csv(args.manifest_out, index=False)
    print(f"Processed pages: {len(df)}")
    print(f"Manifest saved: {args.manifest_out}")


if __name__ == "__main__":
    main()

