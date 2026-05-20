#!/usr/bin/env python3
"""
Segment handwritten line crops from region crops.

Input:
- region_manifest.csv from crop_regions_from_yolo.py

Output:
- line crop images
- context images (line box over region)
- line_manifest.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def detect_line_boxes(region_bgr: np.ndarray, min_line_h: int, merge_gap: int) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 15
    )

    # Connect text components in each line.
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    merged = cv2.dilate(bw, k, iterations=1)

    row_sum = (merged > 0).sum(axis=1)
    threshold = max(8, int(0.08 * merged.shape[1]))

    spans = []
    in_span = False
    y0 = 0
    for y, value in enumerate(row_sum):
        if value > threshold and not in_span:
            in_span = True
            y0 = y
        elif value <= threshold and in_span:
            in_span = False
            y1 = y
            if y1 - y0 >= min_line_h:
                spans.append((y0, y1))
    if in_span:
        y1 = len(row_sum) - 1
        if y1 - y0 >= min_line_h:
            spans.append((y0, y1))

    if not spans:
        return []

    # Merge very close spans.
    merged_spans = [list(spans[0])]
    for a, b in spans[1:]:
        prev = merged_spans[-1]
        if a - prev[1] <= merge_gap:
            prev[1] = b
        else:
            merged_spans.append([a, b])

    line_boxes = []
    for y1, y2 in merged_spans:
        slice_bw = bw[y1:y2, :]
        col_sum = (slice_bw > 0).sum(axis=0)
        xs = np.where(col_sum > 0)[0]
        if xs.size == 0:
            continue
        x1 = max(0, int(xs.min()) - 6)
        x2 = min(bw.shape[1] - 1, int(xs.max()) + 6)
        if x2 - x1 < 40:
            continue
        line_boxes.append((x1, y1, x2, y2))

    return line_boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment line crops from handwritten region crops.")
    parser.add_argument("--region-manifest", type=Path, required=True, help="Input region manifest CSV.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output root for line crops.")
    parser.add_argument("--manifest-out", type=Path, required=True, help="Output line manifest CSV.")
    parser.add_argument("--min-line-height", type=int, default=14, help="Minimum line height.")
    parser.add_argument("--merge-gap", type=int, default=8, help="Merge close horizontal spans.")
    parser.add_argument("--line-padding", type=int, default=4, help="Padding around line crop.")
    parser.add_argument("--min-line-width", type=int, default=60, help="Minimum line width after crop.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lines_dir = args.output_dir / "lines"
    context_dir = args.output_dir / "context"
    lines_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)

    regions = pd.read_csv(args.region_manifest)
    required_cols = {
        "region_id",
        "page_id",
        "region_image_path",
        "page_image_path",
        "x1_page",
        "y1_page",
        "x2_page",
        "y2_page",
    }
    missing = required_cols - set(regions.columns)
    if missing:
        raise ValueError(f"region manifest missing columns: {sorted(missing)}")

    rows = []
    for _, r in tqdm(regions.iterrows(), total=len(regions), desc="Segmenting lines"):
        region_img = cv2.imread(str(r["region_image_path"]))
        if region_img is None:
            continue

        boxes = detect_line_boxes(region_img, min_line_h=args.min_line_height, merge_gap=args.merge_gap)
        if not boxes:
            continue

        context = region_img.copy()
        region_h, region_w = region_img.shape[:2]
        rx1, ry1 = int(r["x1_page"]), int(r["y1_page"])

        line_idx = 0
        emitted_indices = []
        for (x1, y1, x2, y2) in boxes:
            x1p = max(0, x1 - args.line_padding)
            y1p = max(0, y1 - args.line_padding)
            x2p = min(region_w - 1, x2 + args.line_padding)
            y2p = min(region_h - 1, y2 + args.line_padding)

            if x2p - x1p < args.min_line_width:
                continue

            line = region_img[y1p:y2p, x1p:x2p]
            line_id = f"{r['region_id']}_l{line_idx:02d}"
            line_path = lines_dir / f"{line_id}.png"
            cv2.imwrite(str(line_path), line)

            cv2.rectangle(context, (x1p, y1p), (x2p, y2p), (0, 220, 0), 2)

            rows.append(
                {
                    "line_id": line_id,
                    "page_id": r["page_id"],
                    "region_id": r["region_id"],
                    "line_image_path": str(line_path),
                    "region_image_path": str(r["region_image_path"]),
                    "page_image_path": str(r["page_image_path"]),
                    "x1_region": x1p,
                    "y1_region": y1p,
                    "x2_region": x2p,
                    "y2_region": y2p,
                    "x1_page": rx1 + x1p,
                    "y1_page": ry1 + y1p,
                    "x2_page": rx1 + x2p,
                    "y2_page": ry1 + y2p,
                }
            )
            emitted_indices.append(len(rows) - 1)
            line_idx += 1

        context_path = context_dir / f"{r['region_id']}_context.png"
        cv2.imwrite(str(context_path), context)
        for idx in emitted_indices:
            rows[idx]["context_image_path"] = str(context_path)

    out = pd.DataFrame(rows).sort_values(["page_id", "region_id", "line_id"]).reset_index(drop=True)
    out.to_csv(args.manifest_out, index=False)
    print(f"Total line crops: {len(out)}")
    print(f"Line manifest saved: {args.manifest_out}")


if __name__ == "__main__":
    main()
