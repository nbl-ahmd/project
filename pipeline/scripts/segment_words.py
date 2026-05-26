#!/usr/bin/env python3
"""
Segment word crops from line crops.

Input:
- line_manifest.csv from segment_lines.py

Output:
- word crop images
- context images with numbered word boxes over the source line
- optional segmentation overview images for reports/slides
- word_manifest.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from segment_lines import binarize_handwriting, enhance_for_ocr
except ImportError:  # pragma: no cover
    from .segment_lines import binarize_handwriting, enhance_for_ocr


Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class WordSegmentationStats:
    median_component_width: float
    median_component_height: float
    median_component_area: float
    component_count: int
    rejected_tiny_components: int
    adaptive_merge_gap: int
    inter_word_gap: int


def expand_box(box: Box, pad: int, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width - 1, x2 + pad),
        min(height - 1, y2 + pad),
    )


def estimate_word_stats(binary: np.ndarray, fallback_merge_gap: int) -> WordSegmentationStats:
    h, w = binary.shape[:2]
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    widths: list[int] = []
    heights: list[int] = []
    areas: list[int] = []
    rejected = 0
    for label in range(1, num_labels):
        x, y, bw, bh, area = [int(v) for v in stats[label]]
        if area < 5 or bw < 2 or bh < 2:
            rejected += 1
            continue
        if bw > w * 0.92 or bh > h * 0.92:
            rejected += 1
            continue
        widths.append(bw)
        heights.append(bh)
        areas.append(area)

    if not widths:
        return WordSegmentationStats(8.0, 12.0, 40.0, 0, rejected, fallback_merge_gap, fallback_merge_gap * 2)

    median_w = float(np.median(widths))
    median_h = float(np.median(heights))
    median_area = float(np.median(areas))
    adaptive_merge = max(fallback_merge_gap, int(np.clip(median_w * 1.6, 5, 18)))
    inter_word_gap = max(adaptive_merge + 2, int(np.clip(median_w * 2.2, 10, max(12, w // 8))))
    return WordSegmentationStats(median_w, median_h, median_area, len(widths), rejected, adaptive_merge, inter_word_gap)


def raw_component_boxes(binary: np.ndarray, stats: WordSegmentationStats) -> tuple[list[Box], list[Box]]:
    h, w = binary.shape[:2]
    num_labels, _, cc_stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    main: list[Box] = []
    small: list[Box] = []
    min_area = max(5, int(stats.median_component_area * 0.16))
    for label in range(1, num_labels):
        x, y, bw, bh, area = [int(v) for v in cc_stats[label]]
        if bw > w * 0.92 or bh > h * 0.95:
            continue
        box = (x, y, min(w - 1, x + bw), min(h - 1, y + bh))
        if area < min_area or bw < 2 or bh < 2:
            small.append(box)
        else:
            main.append(box)
    return sorted(main, key=lambda b: (b[0], b[1])), sorted(small, key=lambda b: (b[0], b[1]))


def merge_close_components(boxes: list[Box], merge_gap: int) -> list[Box]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[0], b[1]))
    merged: list[list[int]] = [list(boxes[0])]
    for x1, y1, x2, y2 in boxes[1:]:
        prev = merged[-1]
        horizontal_gap = x1 - prev[2]
        vertical_overlap = min(prev[3], y2) - max(prev[1], y1)
        vertical_close = vertical_overlap > -max(4, int((prev[3] - prev[1]) * 0.35))
        if horizontal_gap <= merge_gap and vertical_close:
            prev[0] = min(prev[0], x1)
            prev[1] = min(prev[1], y1)
            prev[2] = max(prev[2], x2)
            prev[3] = max(prev[3], y2)
        else:
            merged.append([x1, y1, x2, y2])
    return [tuple(b) for b in merged]


def attach_small_components(words: list[Box], small_boxes: list[Box], stats: WordSegmentationStats) -> list[Box]:
    if not words:
        return []
    merged = [list(b) for b in words]
    max_distance = max(stats.inter_word_gap, int(stats.median_component_width * 3))
    for sx1, sy1, sx2, sy2 in small_boxes:
        smid_x = (sx1 + sx2) / 2
        smid_y = (sy1 + sy2) / 2
        best_idx = -1
        best_dist = float("inf")
        for idx, word in enumerate(merged):
            wx1, wy1, wx2, wy2 = word
            if smid_x < wx1:
                dx = wx1 - smid_x
            elif smid_x > wx2:
                dx = smid_x - wx2
            else:
                dx = 0
            wmid_y = (wy1 + wy2) / 2
            dy = abs(smid_y - wmid_y)
            dist = dx + dy * 0.35
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx >= 0 and best_dist <= max_distance:
            word = merged[best_idx]
            word[0] = min(word[0], sx1)
            word[1] = min(word[1], sy1)
            word[2] = max(word[2], sx2)
            word[3] = max(word[3], sy2)
    return [tuple(b) for b in merged]


def split_wide_box_by_valleys(box: Box, binary: np.ndarray, stats: WordSegmentationStats) -> list[Box]:
    x1, y1, x2, y2 = box
    width = x2 - x1
    if width < max(stats.median_component_width * 7, stats.inter_word_gap * 2):
        return [box]
    roi = binary[y1:y2, x1:x2]
    if roi.size == 0:
        return [box]
    col_ink = (roi > 0).sum(axis=0)
    whitespace = col_ink <= max(0, int(stats.median_component_height * 0.08))
    spans: list[tuple[int, int]] = []
    start = None
    for idx, is_blank in enumerate(whitespace):
        if is_blank and start is None:
            start = idx
        elif not is_blank and start is not None:
            if idx - start >= max(3, int(stats.median_component_width * 0.8)):
                spans.append((start, idx))
            start = None
    if start is not None and len(whitespace) - start >= max(3, int(stats.median_component_width * 0.8)):
        spans.append((start, len(whitespace) - 1))

    cut_points = [int((a + b) / 2) for a, b in spans if a > stats.median_component_width and width - b > stats.median_component_width]
    if not cut_points:
        return [box]

    parts: list[Box] = []
    left = 0
    for cut in cut_points:
        if cut - left >= stats.median_component_width * 1.5:
            parts.append((x1 + left, y1, x1 + cut, y2))
        left = cut
    if width - left >= stats.median_component_width * 1.5:
        parts.append((x1 + left, y1, x2, y2))
    return parts if len(parts) > 1 else [box]


def filter_word_boxes(boxes: list[Box], min_word_width: int, min_word_height: int, line_shape: tuple[int, int]) -> list[Box]:
    h, w = line_shape
    filtered: list[Box] = []
    for x1, y1, x2, y2 in boxes:
        bw = x2 - x1
        bh = y2 - y1
        if bw < min_word_width or bh < min_word_height:
            continue
        if bw > w * 0.96 or bh > h * 0.96:
            continue
        filtered.append((x1, y1, x2, y2))
    return sorted(filtered, key=lambda b: (b[0], b[1]))


def detect_word_boxes(
    line_bgr: np.ndarray,
    min_word_width: int,
    min_word_height: int,
    merge_gap: int,
) -> list[Box]:
    boxes, _ = detect_word_boxes_with_metadata(line_bgr, min_word_width, min_word_height, merge_gap)
    return boxes


def detect_word_boxes_with_metadata(
    line_bgr: np.ndarray,
    min_word_width: int,
    min_word_height: int,
    merge_gap: int,
) -> tuple[list[Box], dict[str, object]]:
    binary = binarize_handwriting(line_bgr)
    h, w = binary.shape[:2]
    stats = estimate_word_stats(binary, merge_gap)

    kernel_w = int(np.clip(stats.median_component_width * 1.8, 7, 18))
    kernel_h = int(np.clip(stats.median_component_height * 0.16, 1, 3))
    merged_img = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h)), iterations=1)
    contours, _ = cv2.findContours(merged_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_boxes = sorted(
        [(x, y, min(w - 1, x + bw), min(h - 1, y + bh)) for x, y, bw, bh in (cv2.boundingRect(c) for c in contours)],
        key=lambda b: (b[0], b[1]),
    )
    contour_boxes = [b for b in contour_boxes if (b[2] - b[0]) >= 2 and (b[3] - b[1]) >= 2]

    main_components, small_components = raw_component_boxes(binary, stats)
    candidates = contour_boxes if len(contour_boxes) >= max(1, len(main_components) // 3) else main_components
    words = merge_close_components(candidates, stats.adaptive_merge_gap)
    words = attach_small_components(words, small_components, stats)

    refined: list[Box] = []
    for box in words:
        refined.extend(split_wide_box_by_valleys(box, binary, stats))
    words = filter_word_boxes(refined, min_word_width, min_word_height, binary.shape[:2])

    metadata: dict[str, object] = {
        "method": "hybrid_components_gap_valley",
        "component_count": stats.component_count,
        "rejected_tiny_components": stats.rejected_tiny_components,
        "median_component_width": round(stats.median_component_width, 3),
        "median_component_height": round(stats.median_component_height, 3),
        "adaptive_merge_gap": stats.adaptive_merge_gap,
        "inter_word_gap": stats.inter_word_gap,
        "quality_flag": "ok" if words else "no_words_detected",
    }
    return words, metadata


def draw_numbered_boxes(image_bgr: np.ndarray, boxes: list[Box], color: tuple[int, int, int]) -> np.ndarray:
    canvas = image_bgr.copy()
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, str(idx), (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return canvas


def render_word_overview(line_bgr: np.ndarray, binary: np.ndarray, boxes: list[Box], output_path: Path) -> None:
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    boxed = draw_numbered_boxes(line_bgr, boxes, (0, 0, 230))
    h = max(line_bgr.shape[0], binary_bgr.shape[0], boxed.shape[0])
    panels = []
    for panel in [line_bgr, binary_bgr, boxed]:
        if panel.shape[0] != h:
            scale = h / panel.shape[0]
            panel = cv2.resize(panel, (int(panel.shape[1] * scale), h), interpolation=cv2.INTER_AREA)
        panels.append(panel)
    overview = cv2.hconcat(panels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overview)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment word crops from line crops.")
    parser.add_argument("--line-manifest", type=Path, required=True, help="Input line_manifest.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output root for word crops")
    parser.add_argument("--manifest-out", type=Path, required=True, help="Output word_manifest.csv")
    parser.add_argument("--review-out", type=Path, default=None, help="Optional segmentation_review.csv output.")
    parser.add_argument("--overview-dir", type=Path, default=None, help="Optional folder for paper-friendly overview images.")
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

    rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    for _, row in tqdm(lines.iterrows(), total=len(lines), desc="Segmenting words"):
        line_path = Path(str(row["line_image_path"]))
        line_bgr = cv2.imread(str(line_path))
        if line_bgr is None:
            continue

        boxes, metadata = detect_word_boxes_with_metadata(
            line_bgr,
            min_word_width=args.min_word_width,
            min_word_height=args.min_word_height,
            merge_gap=args.merge_gap,
        )
        if not boxes:
            review_rows.append(
                {
                    "page_id": row["page_id"],
                    "region_id": row["region_id"],
                    "line_id": row["line_id"],
                    "stage": "word",
                    "detected_count": 0,
                    "method": metadata["method"],
                    "quality_flag": metadata["quality_flag"],
                    "rejected_tiny_components": metadata["rejected_tiny_components"],
                    "notes": "no word boxes emitted",
                }
            )
            continue

        context = line_bgr.copy()
        line_h, line_w = line_bgr.shape[:2]
        lx1, ly1 = int(row["x1_page"]), int(row["y1_page"])

        emitted_boxes: list[Box] = []
        for word_idx, box in enumerate(boxes):
            x1p, y1p, x2p, y2p = expand_box(box, args.word_padding, line_w, line_h)
            if x2p - x1p < args.min_word_width or y2p - y1p < args.min_word_height:
                continue

            word_id = f"{row['line_id']}_w{word_idx:02d}"
            word_path = words_dir / f"{word_id}.png"
            cv2.imwrite(str(word_path), enhance_for_ocr(line_bgr[y1p:y2p, x1p:x2p]))
            cv2.rectangle(context, (x1p, y1p), (x2p, y2p), (0, 0, 230), 2)
            cv2.putText(context, str(word_idx), (x1p, max(14, y1p - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 230), 2)
            emitted_boxes.append((x1p, y1p, x2p, y2p))

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
                    "segmentation_method": metadata["method"],
                    "quality_flag": metadata["quality_flag"],
                    "median_component_width": metadata["median_component_width"],
                    "adaptive_merge_gap": metadata["adaptive_merge_gap"],
                }
            )

        context_path = context_dir / f"{row['line_id']}_word_context.png"
        cv2.imwrite(str(context_path), context)
        if args.overview_dir is not None:
            overview_path = args.overview_dir / f"{row['line_id']}_word_overview.png"
            render_word_overview(line_bgr, binarize_handwriting(line_bgr), emitted_boxes, overview_path)

        review_rows.append(
            {
                "page_id": row["page_id"],
                "region_id": row["region_id"],
                "line_id": row["line_id"],
                "stage": "word",
                "detected_count": len(emitted_boxes),
                "method": metadata["method"],
                "quality_flag": "ok" if emitted_boxes else "no_words_emitted",
                "rejected_tiny_components": metadata["rejected_tiny_components"],
                "notes": f"merge_gap={metadata['adaptive_merge_gap']};inter_word_gap={metadata['inter_word_gap']}",
            }
        )

    out = pd.DataFrame(rows).sort_values(["page_id", "region_id", "line_id", "word_id"]).reset_index(drop=True)
    out.to_csv(args.manifest_out, index=False)
    if args.review_out is not None:
        args.review_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(review_rows).to_csv(args.review_out, index=False)
    print(f"Total word crops: {len(out)}")
    print(f"Word manifest saved: {args.manifest_out}")


if __name__ == "__main__":
    main()
