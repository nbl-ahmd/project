#!/usr/bin/env python3
"""
Segment word crops from line crops.

Input:
- line_manifest.csv from segment_lines.py

Output:
- word crop images
- context images with word boxes over the source line
- word_manifest.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def detect_word_boxes(
    line_bgr: np.ndarray,
    min_word_width: int,
    min_word_height: int,
    merge_gap: int,
) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(line_bgr, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 15
    )

    # Light horizontal dilation joins letters inside a word while preserving gaps.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 2))
    merged = cv2.dilate(bw, kernel, iterations=1)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    h, w = bw.shape[:2]
    for contour in contours:
        x, y, bwid, bhei = cv2.boundingRect(contour)
        if bwid < min_word_width or bhei < min_word_height:
            continue
        if bwid * bhei < 60:
            continue
        boxes.append((x, y, min(w - 1, x + bwid), min(h - 1, y + bhei)))

    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b[0])
    merged_boxes: list[list[int]] = [list(boxes[0])]
    for x1, y1, x2, y2 in boxes[1:]:
        prev = merged_boxes[-1]
        if x1 - prev[2] <= merge_gap:
            prev[0] = min(prev[0], x1)
            prev[1] = min(prev[1], y1)
            prev[2] = max(prev[2], x2)
            prev[3] = max(prev[3], y2)
        else:
            merged_boxes.append([x1, y1, x2, y2])

    return [tuple(b) for b in merged_boxes if b[2] - b[0] >= min_word_width]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment word crops from line crops.")
    parser.add_argument("--line-manifest", type=Path, required=True, help="Input line_manifest.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output root for word crops")
    parser.add_argument("--manifest-out", type=Path, required=True, help="Output word_manifest.csv")
    parser.add_argument("--min-word-width", type=int, default=16)
    parser.add_argument("--min-word-height", type=int, default=8)
    parser.add_argument("--merge-gap", type=int, default=8)
    parser.add_argument("--word-padding", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    words_dir = args.output_dir / "words"
    context_dir = args.output_dir / "context"
    words_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)

    lines = pd.read_csv(args.line_manifest)
    required = {
        "line_id",
        "page_id",
        "region_id",
        "line_image_path",
        "x1_page",
        "y1_page",
        "x2_page",
        "y2_page",
    }
    missing = required - set(lines.columns)
    if missing:
        raise ValueError(f"line manifest missing columns: {sorted(missing)}")

    rows = []
    for _, row in tqdm(lines.iterrows(), total=len(lines), desc="Segmenting words"):
        line_path = Path(str(row["line_image_path"]))
        line_bgr = cv2.imread(str(line_path))
        if line_bgr is None:
            continue

        boxes = detect_word_boxes(
            line_bgr,
            min_word_width=args.min_word_width,
            min_word_height=args.min_word_height,
            merge_gap=args.merge_gap,
        )
        if not boxes:
            continue

        context = line_bgr.copy()
        line_h, line_w = line_bgr.shape[:2]
        lx1, ly1 = int(row["x1_page"]), int(row["y1_page"])

        for word_idx, (x1, y1, x2, y2) in enumerate(boxes):
            x1p = max(0, x1 - args.word_padding)
            y1p = max(0, y1 - args.word_padding)
            x2p = min(line_w - 1, x2 + args.word_padding)
            y2p = min(line_h - 1, y2 + args.word_padding)

            word_id = f"{row['line_id']}_w{word_idx:02d}"
            word_path = words_dir / f"{word_id}.png"
            cv2.imwrite(str(word_path), line_bgr[y1p:y2p, x1p:x2p])
            cv2.rectangle(context, (x1p, y1p), (x2p, y2p), (0, 120, 255), 2)

            rows.append(
                {
                    "word_id": word_id,
                    "line_id": row["line_id"],
                    "region_id": row["region_id"],
                    "page_id": row["page_id"],
                    "word_image_path": str(word_path),
                    "line_image_path": str(line_path),
                    "line_context_image_path": str(context_dir / f"{row['line_id']}_word_context.png"),
                    "x1_line": x1p,
                    "y1_line": y1p,
                    "x2_line": x2p,
                    "y2_line": y2p,
                    "x1_page": lx1 + x1p,
                    "y1_page": ly1 + y1p,
                    "x2_page": lx1 + x2p,
                    "y2_page": ly1 + y2p,
                }
            )

        cv2.imwrite(str(context_dir / f"{row['line_id']}_word_context.png"), context)

    out = pd.DataFrame(rows).sort_values(["page_id", "region_id", "line_id", "word_id"]).reset_index(drop=True)
    out.to_csv(args.manifest_out, index=False)
    print(f"Total word crops: {len(out)}")
    print(f"Word manifest saved: {args.manifest_out}")


if __name__ == "__main__":
    main()

