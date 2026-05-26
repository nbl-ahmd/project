#!/usr/bin/env python3
"""
Segment handwritten line crops from region crops.

Input:
- region_manifest.csv from crop_regions_from_yolo.py or run_end_to_end.py

Output:
- line crop images
- context images with numbered line boxes over each region
- optional segmentation overview images for reports/slides
- line_manifest.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class SegmentationStats:
    median_component_width: float
    median_component_height: float
    median_component_area: float
    component_count: int
    rejected_tiny_components: int


def remove_form_rulings(binary: np.ndarray) -> np.ndarray:
    """Remove long prescription-form lines while keeping handwriting strokes."""
    h, w = binary.shape[:2]
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, h // 3)))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(80, w // 3), 1))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    rulings = cv2.bitwise_or(vertical, horizontal)
    return cv2.bitwise_and(binary, cv2.bitwise_not(rulings))


def illumination_compensated_gray(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    bg_kernel = max(31, (min(gray.shape[:2]) // 8) * 2 + 1)
    background = cv2.GaussianBlur(gray, (bg_kernel, bg_kernel), 0)
    normalized = cv2.divide(gray, background, scale=255)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(normalized)


def enhance_for_ocr(image_bgr: np.ndarray) -> np.ndarray:
    """Return an illumination-normalized RGB crop for OCR/model input."""
    enhanced = illumination_compensated_gray(image_bgr)
    enhanced = cv2.fastNlMeansDenoising(enhanced, None, h=8, templateWindowSize=7, searchWindowSize=21)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def binarize_handwriting(region_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    compensated = illumination_compensated_gray(region_bgr)
    blur = cv2.GaussianBlur(compensated, (3, 3), 0)
    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12
    )
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.bitwise_or(adaptive, otsu)
    binary = remove_form_rulings(binary)
    border = cv2.inRange(gray, 0, 25)
    binary = cv2.bitwise_and(binary, cv2.bitwise_not(border))
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, noise_kernel, iterations=1)


def estimate_component_stats(binary: np.ndarray) -> SegmentationStats:
    h, w = binary.shape[:2]
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    widths: list[int] = []
    heights: list[int] = []
    areas: list[int] = []
    rejected = 0
    for label in range(1, num_labels):
        x, y, bw, bh, area = [int(v) for v in stats[label]]
        if area < 6 or bw < 2 or bh < 2:
            rejected += 1
            continue
        if bw > w * 0.92 or bh > h * 0.35:
            rejected += 1
            continue
        widths.append(bw)
        heights.append(bh)
        areas.append(area)

    if not heights:
        return SegmentationStats(8.0, 12.0, 40.0, 0, rejected)
    return SegmentationStats(
        median_component_width=float(np.median(widths)),
        median_component_height=float(np.median(heights)),
        median_component_area=float(np.median(areas)),
        component_count=len(heights),
        rejected_tiny_components=rejected,
    )


def expand_box(box: Box, pad_x: int, pad_y: int, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width - 1, x2 + pad_x),
        min(height - 1, y2 + pad_y),
    )


def merge_overlapping_boxes(boxes: list[Box], merge_gap: int) -> list[Box]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged: list[list[int]] = [list(boxes[0])]
    for x1, y1, x2, y2 in boxes[1:]:
        prev = merged[-1]
        prev_mid = (prev[1] + prev[3]) / 2
        mid = (y1 + y2) / 2
        overlap = min(prev[3], y2) - max(prev[1], y1)
        same_band = overlap > 0 or abs(mid - prev_mid) <= max(merge_gap, (prev[3] - prev[1]) * 0.65)
        if same_band:
            prev[0] = min(prev[0], x1)
            prev[1] = min(prev[1], y1)
            prev[2] = max(prev[2], x2)
            prev[3] = max(prev[3], y2)
        else:
            merged.append([x1, y1, x2, y2])
    return [tuple(b) for b in merged]


def component_boxes(binary: np.ndarray, stats_hint: SegmentationStats) -> list[Box]:
    h, w = binary.shape[:2]
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: list[Box] = []
    min_area = max(6, int(stats_hint.median_component_area * 0.18))
    for label in range(1, num_labels):
        x, y, bw, bh, area = [int(v) for v in stats[label]]
        if area < min_area or bh < 2 or bw < 2:
            continue
        if bh > h * 0.32 or bw > w * 0.94:
            continue
        boxes.append((x, y, x + bw, y + bh))
    return boxes


def detect_line_boxes_connected(binary: np.ndarray, min_line_h: int, merge_gap: int, stats: SegmentationStats) -> list[Box]:
    h, w = binary.shape[:2]
    kernel_w = int(np.clip(stats.median_component_width * 5.5, 24, max(28, w // 9)))
    kernel_h = int(np.clip(stats.median_component_height * 0.35, 2, 8))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    merged = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[Box] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bh < min_line_h or bw < max(40, int(w * 0.06)):
            continue
        if bh > h * 0.38 or bw > w * 0.99:
            continue
        ink = binary[y : y + bh, x : x + bw]
        if cv2.countNonZero(ink) < max(20, 0.0018 * bw * bh):
            continue
        boxes.append((x, y, x + bw, y + bh))

    return sorted(merge_overlapping_boxes(boxes, merge_gap=merge_gap), key=lambda b: (b[1], b[0]))


def detect_line_boxes_projection(binary: np.ndarray, min_line_h: int, merge_gap: int, stats: SegmentationStats) -> list[Box]:
    h, w = binary.shape[:2]
    kernel_w = int(np.clip(stats.median_component_width * 4.0, 18, 70))
    kernel_h = int(np.clip(stats.median_component_height * 0.25, 2, 6))
    merged = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h)), iterations=1)

    row_sum = (merged > 0).sum(axis=1)
    nonzero_rows = row_sum[row_sum > 0]
    if nonzero_rows.size == 0:
        return []
    threshold = max(2, int(np.percentile(nonzero_rows, 28)))

    spans: list[tuple[int, int]] = []
    in_span = False
    y0 = 0
    for y, value in enumerate(row_sum):
        if value > threshold and not in_span:
            in_span = True
            y0 = y
        elif value <= threshold and in_span:
            in_span = False
            if y - y0 >= min_line_h:
                spans.append((y0, y))
    if in_span and len(row_sum) - y0 >= min_line_h:
        spans.append((y0, len(row_sum) - 1))

    if not spans:
        return []

    merged_spans: list[list[int]] = [list(spans[0])]
    adaptive_gap = max(merge_gap, int(stats.median_component_height * 0.35))
    for a, b in spans[1:]:
        prev = merged_spans[-1]
        if a - prev[1] <= adaptive_gap:
            prev[1] = b
        else:
            merged_spans.append([a, b])

    line_boxes: list[Box] = []
    for y1, y2 in merged_spans:
        slice_bw = binary[y1:y2, :]
        col_sum = (slice_bw > 0).sum(axis=0)
        xs = np.where(col_sum > 0)[0]
        if xs.size == 0:
            continue
        x1 = max(0, int(xs.min()) - 6)
        x2 = min(w - 1, int(xs.max()) + 6)
        if x2 - x1 < max(40, int(w * 0.06)):
            continue
        line_boxes.append((x1, max(0, y1 - 1), x2, min(h - 1, y2 + 1)))

    return line_boxes


def detect_line_boxes_components(binary: np.ndarray, min_line_h: int, merge_gap: int, stats: SegmentationStats) -> list[Box]:
    h, w = binary.shape[:2]
    components = component_boxes(binary, stats)
    if not components:
        return []

    components = sorted(components, key=lambda b: ((b[1] + b[3]) / 2, b[0]))
    bands: list[list[int]] = []
    tolerance = max(merge_gap + 3, int(stats.median_component_height * 1.1))
    for x1, y1, x2, y2 in components:
        mid = (y1 + y2) / 2
        placed = False
        for band in bands:
            band_mid = (band[1] + band[3]) / 2
            band_h = max(1, band[3] - band[1])
            if abs(mid - band_mid) <= max(tolerance, int(band_h * 0.55)):
                band[0] = min(band[0], x1)
                band[1] = min(band[1], y1)
                band[2] = max(band[2], x2)
                band[3] = max(band[3], y2)
                placed = True
                break
        if not placed:
            bands.append([x1, y1, x2, y2])

    boxes: list[Box] = []
    for x1, y1, x2, y2 in bands:
        if y2 - y1 < min_line_h or x2 - x1 < max(40, int(w * 0.06)):
            continue
        boxes.append(expand_box((x1, y1, x2, y2), 8, 2, w, h))
    return sorted(merge_overlapping_boxes(boxes, merge_gap=merge_gap), key=lambda b: (b[1], b[0]))


def refine_line_separators(binary: np.ndarray, boxes: list[Box], stats: SegmentationStats) -> list[Box]:
    """Use low-ink valleys between adjacent bands as a lightweight seam-carving analogue."""
    if len(boxes) < 2:
        return boxes
    h, w = binary.shape[:2]
    row_ink = (binary > 0).sum(axis=1).astype(np.float32)
    refined = [list(b) for b in sorted(boxes, key=lambda b: (b[1], b[0]))]
    min_gap = max(2, int(stats.median_component_height * 0.18))
    min_line_h = max(6, int(stats.median_component_height * 0.75))

    for i in range(len(refined) - 1):
        upper = refined[i]
        lower = refined[i + 1]
        search_top = max(0, upper[1] + min_line_h)
        search_bottom = min(h - 1, lower[3] - min_line_h)
        if search_bottom <= search_top:
            continue
        valley = int(search_top + np.argmin(row_ink[search_top : search_bottom + 1]))
        upper[3] = min(upper[3], max(upper[1] + min_line_h, valley - min_gap))
        lower[1] = max(lower[1], min(lower[3] - min_line_h, valley + min_gap))

    cleaned: list[Box] = []
    for x1, y1, x2, y2 in refined:
        if x2 > x1 and y2 > y1:
            cleaned.append((x1, y1, x2, y2))
    return cleaned


def choose_line_candidates(candidates: list[tuple[str, list[Box]]], region_shape: tuple[int, int]) -> tuple[str, list[Box]]:
    valid = [(name, boxes) for name, boxes in candidates if boxes]
    if not valid:
        return "none", []
    h, _ = region_shape

    def score(item: tuple[str, list[Box]]) -> tuple[float, float, float]:
        _, boxes = item
        heights = np.array([max(1, b[3] - b[1]) for b in boxes], dtype=np.float32)
        coverage = float(sum(heights) / max(1, h))
        giant_penalty = float(np.sum(heights > h * 0.32))
        # More plausible rows are good; huge boxes and very uneven heights are bad.
        return (len(boxes) - giant_penalty * 2.0, -float(np.std(heights)), -abs(coverage - 0.32))

    return max(valid, key=score)


def detect_line_boxes(region_bgr: np.ndarray, min_line_h: int, merge_gap: int) -> list[Box]:
    binary = binarize_handwriting(region_bgr)
    stats = estimate_component_stats(binary)
    adaptive_min_h = max(min_line_h, int(stats.median_component_height * 0.75))
    adaptive_gap = max(merge_gap, int(stats.median_component_height * 0.45))

    projection = detect_line_boxes_projection(binary, adaptive_min_h, adaptive_gap, stats)
    components = detect_line_boxes_components(binary, adaptive_min_h, adaptive_gap, stats)
    connected = detect_line_boxes_connected(binary, adaptive_min_h, adaptive_gap, stats)
    method, boxes = choose_line_candidates(
        [("projection", projection), ("components", components), ("connected", connected)],
        binary.shape[:2],
    )
    if not boxes:
        return []
    refined = refine_line_separators(binary, boxes, stats)
    return sorted(refined if refined else boxes, key=lambda b: (b[1], b[0]))


def detect_line_boxes_with_metadata(region_bgr: np.ndarray, min_line_h: int, merge_gap: int) -> tuple[list[Box], dict[str, object]]:
    binary = binarize_handwriting(region_bgr)
    stats = estimate_component_stats(binary)
    adaptive_min_h = max(min_line_h, int(stats.median_component_height * 0.75))
    adaptive_gap = max(merge_gap, int(stats.median_component_height * 0.45))
    candidates = [
        ("projection", detect_line_boxes_projection(binary, adaptive_min_h, adaptive_gap, stats)),
        ("components", detect_line_boxes_components(binary, adaptive_min_h, adaptive_gap, stats)),
        ("connected", detect_line_boxes_connected(binary, adaptive_min_h, adaptive_gap, stats)),
    ]
    method, boxes = choose_line_candidates(candidates, binary.shape[:2])
    refined = refine_line_separators(binary, boxes, stats) if boxes else []
    final_boxes = sorted(refined if refined else boxes, key=lambda b: (b[1], b[0]))
    heights = [b[3] - b[1] for b in final_boxes]
    metadata: dict[str, object] = {
        "method": f"hybrid_{method}_seam_refined" if final_boxes else "none",
        "component_count": stats.component_count,
        "rejected_tiny_components": stats.rejected_tiny_components,
        "median_component_width": round(stats.median_component_width, 3),
        "median_component_height": round(stats.median_component_height, 3),
        "median_line_height": round(float(np.median(heights)), 3) if heights else 0.0,
        "quality_flag": "ok" if final_boxes else "no_lines_detected",
        "candidate_counts": ";".join(f"{name}:{len(candidate_boxes)}" for name, candidate_boxes in candidates),
    }
    return final_boxes, metadata


def draw_numbered_boxes(image_bgr: np.ndarray, boxes: list[Box], color: tuple[int, int, int], prefix: str = "") -> np.ndarray:
    canvas = image_bgr.copy()
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{prefix}{idx}"
        y_text = max(14, y1 - 5)
        cv2.putText(canvas, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return canvas


def render_line_overview(region_bgr: np.ndarray, binary: np.ndarray, boxes: list[Box], output_path: Path) -> None:
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    boxed = draw_numbered_boxes(region_bgr, boxes, (0, 180, 0), prefix="L")
    h = max(region_bgr.shape[0], binary_bgr.shape[0], boxed.shape[0])
    panels = []
    for panel in [region_bgr, binary_bgr, boxed]:
        if panel.shape[0] != h:
            scale = h / panel.shape[0]
            panel = cv2.resize(panel, (int(panel.shape[1] * scale), h), interpolation=cv2.INTER_AREA)
        panels.append(panel)
    overview = cv2.hconcat(panels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overview)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment line crops from handwritten region crops.")
    parser.add_argument("--region-manifest", type=Path, required=True, help="Input region manifest CSV.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output root for line crops.")
    parser.add_argument("--manifest-out", type=Path, required=True, help="Output line manifest CSV.")
    parser.add_argument("--review-out", type=Path, default=None, help="Optional segmentation_review.csv output.")
    parser.add_argument("--overview-dir", type=Path, default=None, help="Optional folder for paper-friendly overview images.")
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
        "x1_page",
        "y1_page",
        "x2_page",
        "y2_page",
    }
    missing = required_cols - set(regions.columns)
    if missing:
        raise ValueError(f"region manifest missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    for _, r in tqdm(regions.iterrows(), total=len(regions), desc="Segmenting lines"):
        region_img = cv2.imread(str(r["region_image_path"]))
        if region_img is None:
            continue

        boxes, metadata = detect_line_boxes_with_metadata(region_img, args.min_line_height, args.merge_gap)
        if not boxes:
            review_rows.append(
                {
                    "page_id": r["page_id"],
                    "region_id": r["region_id"],
                    "stage": "line",
                    "detected_count": 0,
                    "method": metadata["method"],
                    "quality_flag": metadata["quality_flag"],
                    "rejected_tiny_components": metadata["rejected_tiny_components"],
                    "notes": "no line boxes emitted",
                }
            )
            continue

        context = region_img.copy()
        region_h, region_w = region_img.shape[:2]
        rx1, ry1 = int(r["x1_page"]), int(r["y1_page"])

        line_idx = 0
        emitted_indices: list[int] = []
        emitted_boxes: list[Box] = []
        for x1, y1, x2, y2 in boxes:
            x1p, y1p, x2p, y2p = expand_box((x1, y1, x2, y2), args.line_padding, args.line_padding, region_w, region_h)
            if x2p - x1p < args.min_line_width:
                continue

            line = region_img[y1p:y2p, x1p:x2p]
            line_id = f"{r['region_id']}_l{line_idx:02d}"
            line_path = lines_dir / f"{line_id}.png"
            cv2.imwrite(str(line_path), enhance_for_ocr(line))

            cv2.rectangle(context, (x1p, y1p), (x2p, y2p), (0, 220, 0), 2)
            cv2.putText(context, str(line_idx), (x1p, max(15, y1p - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 160, 0), 2)

            rows.append(
                {
                    "line_id": line_id,
                    "page_id": r["page_id"],
                    "region_id": r["region_id"],
                    "line_image_path": str(line_path),
                    "region_image_path": str(r["region_image_path"]),
                    "page_image_path": str(r.get("page_image_path", "")),
                    "x1_region": x1p,
                    "y1_region": y1p,
                    "x2_region": x2p,
                    "y2_region": y2p,
                    "x1_page": rx1 + x1p,
                    "y1_page": ry1 + y1p,
                    "x2_page": rx1 + x2p,
                    "y2_page": ry1 + y2p,
                    "segmentation_method": metadata["method"],
                    "quality_flag": metadata["quality_flag"],
                    "median_component_height": metadata["median_component_height"],
                    "line_height": y2p - y1p,
                }
            )
            emitted_indices.append(len(rows) - 1)
            emitted_boxes.append((x1p, y1p, x2p, y2p))
            line_idx += 1

        context_path = context_dir / f"{r['region_id']}_context.png"
        cv2.imwrite(str(context_path), context)
        for idx in emitted_indices:
            rows[idx]["context_image_path"] = str(context_path)

        if args.overview_dir is not None:
            overview_path = args.overview_dir / f"{r['region_id']}_line_overview.png"
            render_line_overview(region_img, binarize_handwriting(region_img), emitted_boxes, overview_path)

        review_rows.append(
            {
                "page_id": r["page_id"],
                "region_id": r["region_id"],
                "stage": "line",
                "detected_count": line_idx,
                "method": metadata["method"],
                "quality_flag": "ok" if line_idx else "no_lines_emitted",
                "rejected_tiny_components": metadata["rejected_tiny_components"],
                "notes": metadata["candidate_counts"],
            }
        )

    out = pd.DataFrame(rows).sort_values(["page_id", "region_id", "line_id"]).reset_index(drop=True)
    out.to_csv(args.manifest_out, index=False)
    if args.review_out is not None:
        args.review_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(review_rows).to_csv(args.review_out, index=False)
    print(f"Total line crops: {len(out)}")
    print(f"Line manifest saved: {args.manifest_out}")


if __name__ == "__main__":
    main()
