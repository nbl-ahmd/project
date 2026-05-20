#!/usr/bin/env python3
"""
Run an end-to-end prescription recognition demo:

raw page -> preprocessing -> handwritten-region proposal -> line segmentation
-> OCR -> drug/dosage/frequency validation -> CSV/JSON outputs.

The OCR stage can use a local/fine-tuned TrOCR checkpoint, a Hugging Face model
name, or demo text supplied from a file. Demo text is useful for validating the
non-OCR stages on machines without GPU dependencies.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from preprocess_pages import preprocess_page  # noqa: E402
from segment_lines import detect_line_boxes  # noqa: E402
from segment_words import detect_word_boxes  # noqa: E402


VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DOSAGE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|gm|ml|iu|unit|units|tab|tabs|tablet|cap|caps|drop|drops|sachet)s?\b",
    re.IGNORECASE,
)
FREQUENCY_RE = re.compile(
    r"\b(?:\d\s*-\s*\d\s*-\s*\d|od|bd|tds|qid|hs|sos|prn|daily|night|morning|evening|weekly|once|twice)\b",
    re.IGNORECASE,
)


@dataclass
class Region:
    page_id: str
    region_id: str
    image_path: Path
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class LineCrop:
    page_id: str
    region_id: str
    line_id: str
    image_path: Path
    context_path: Path
    x1_page: int
    y1_page: int
    x2_page: int
    y2_page: int


class OCRBackend:
    def predict(self, image_path: Path) -> str:
        raise NotImplementedError


class EmptyOCR(OCRBackend):
    def predict(self, image_path: Path) -> str:
        return ""


class DemoTextOCR(OCRBackend):
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.idx = 0

    def predict(self, image_path: Path) -> str:
        if not self.texts:
            return ""
        value = self.texts[min(self.idx, len(self.texts) - 1)]
        self.idx += 1
        return value


class TrOCROCR(OCRBackend):
    def __init__(self, model_name_or_path: str, max_length: int, num_beams: int) -> None:
        import torch
        from PIL import Image
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.torch = torch
        self.image_cls = Image
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = TrOCRProcessor.from_pretrained(model_name_or_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.num_beams = num_beams

    def predict(self, image_path: Path) -> str:
        image = self.image_cls.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        with self.torch.no_grad():
            ids = self.model.generate(
                pixel_values,
                max_length=self.max_length,
                num_beams=self.num_beams,
                early_stopping=self.num_beams > 1,
            )
        return normalize_text(self.processor.batch_decode(ids, skip_special_tokens=True)[0])


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().split())


def list_images(path: Path) -> Iterable[Path]:
    if path.is_file() and path.suffix.lower() in VALID_EXTS:
        yield path
        return
    for child in sorted(path.rglob("*")):
        if child.is_file() and child.suffix.lower() in VALID_EXTS:
            yield child


def safe_id(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_lexicon(path: Path) -> list[str]:
    values: list[str] = []
    if not path.exists():
        return values
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            candidates = [c for c in ["medicine_name", "MEDICINE_NAME", "drug", "name"] if c in (reader.fieldnames or [])]
            key = candidates[0] if candidates else None
            if key:
                values.extend(normalize_text(row.get(key, "")) for row in reader)
    else:
        values.extend(normalize_text(x) for x in path.read_text(encoding="utf-8").splitlines())
    return sorted({x for x in values if x})


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def best_lexicon_match(text: str, lexicon: list[str]) -> tuple[str, float]:
    if not text or not lexicon:
        return "", 0.0

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]*", text)
    candidates = [text]
    candidates.extend(tokens)
    if len(tokens) >= 2:
        candidates.extend(" ".join(tokens[i : i + 2]) for i in range(len(tokens) - 1))

    best_name = ""
    best_score = 0.0
    for candidate in candidates:
        for drug in lexicon:
            score = similarity(candidate, drug)
            if score > best_score:
                best_name = drug
                best_score = score
    return best_name, round(best_score, 4)


def parse_medical_entities(text: str, lexicon: list[str], threshold: float) -> dict[str, str | float]:
    normalized = normalize_text(text)
    drug, score = best_lexicon_match(normalized, lexicon)
    dosages = [normalize_text(x) for x in DOSAGE_RE.findall(normalized)]
    frequencies = [normalize_text(x).upper() for x in FREQUENCY_RE.findall(normalized)]
    return {
        "ocr_text": normalized,
        "medicine_name": drug if score >= threshold else "",
        "medicine_match_score": score,
        "dosage": "; ".join(dict.fromkeys(dosages)),
        "frequency": "; ".join(dict.fromkeys(frequencies)),
        "validation_status": "matched" if score >= threshold else "needs_review",
    }


def proposal_region_box(page_bgr: np.ndarray) -> tuple[int, int, int, int]:
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    merged = cv2.dilate(binary, kernel, iterations=1)
    h, w = merged.shape[:2]
    mask = np.zeros_like(merged)
    mask[int(h * 0.12) : int(h * 0.9), :] = merged[int(h * 0.12) : int(h * 0.9), :]
    coords = cv2.findNonZero(mask)
    if coords is None:
        return 0, int(h * 0.15), w - 1, int(h * 0.85)
    x, y, bw, bh = cv2.boundingRect(coords)
    pad = 16
    return max(0, x - pad), max(0, y - pad), min(w - 1, x + bw + pad), min(h - 1, y + bh + pad)


def read_yolo_regions(label_path: Path, image_shape: tuple[int, int], target_class: int) -> list[tuple[int, int, int, int]]:
    if not label_path.exists():
        return []
    img_h, img_w = image_shape
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5 or int(float(parts[0])) != target_class:
            continue
        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = int((cx - bw / 2) * img_w)
        y1 = int((cy - bh / 2) * img_h)
        x2 = int((cx + bw / 2) * img_w)
        y2 = int((cy + bh / 2) * img_h)
        boxes.append((max(0, x1), max(0, y1), min(img_w - 1, x2), min(img_h - 1, y2)))
    return boxes


def predict_yolo_regions(
    model,
    image_path: Path,
    target_class: int,
    conf: float,
) -> list[tuple[int, int, int, int]]:
    results = model.predict(str(image_path), conf=conf, verbose=False)
    boxes: list[tuple[int, int, int, int]] = []
    if not results:
        return boxes
    result = results[0]
    if result.boxes is None:
        return boxes
    for box in result.boxes:
        cls_id = int(box.cls[0].item()) if box.cls is not None else 0
        if cls_id != target_class:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes


def create_regions(
    page_path: Path,
    page_bgr: np.ndarray,
    output_dir: Path,
    labels_dir: Path | None,
    target_class: int,
    use_full_page: bool,
    yolo_model=None,
    yolo_conf: float = 0.25,
) -> list[Region]:
    page_id = safe_id(page_path)
    boxes: list[tuple[int, int, int, int]]
    if use_full_page:
        h, w = page_bgr.shape[:2]
        boxes = [(0, 0, w - 1, h - 1)]
    elif yolo_model is not None:
        boxes = predict_yolo_regions(yolo_model, page_path, target_class, yolo_conf)
        if not boxes:
            boxes = [proposal_region_box(page_bgr)]
    elif labels_dir:
        boxes = read_yolo_regions(labels_dir / f"{page_path.stem}.txt", page_bgr.shape[:2], target_class)
        if not boxes:
            boxes = [proposal_region_box(page_bgr)]
    else:
        boxes = [proposal_region_box(page_bgr)]

    regions = []
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        if x2 <= x1 or y2 <= y1:
            continue
        region_id = f"{page_id}_r{idx:02d}"
        region_path = output_dir / f"{region_id}.png"
        cv2.imwrite(str(region_path), page_bgr[y1:y2, x1:x2])
        regions.append(Region(page_id, region_id, region_path, x1, y1, x2, y2))
    return regions


def create_line_crops(
    region: Region,
    output_dir: Path,
    min_line_height: int,
    merge_gap: int,
    line_yolo_model=None,
    line_target_class: int = 0,
    line_yolo_conf: float = 0.25,
) -> list[LineCrop]:
    region_bgr = cv2.imread(str(region.image_path))
    if region_bgr is None:
        return []
    if line_yolo_model is not None:
        boxes = predict_yolo_regions(line_yolo_model, region.image_path, line_target_class, line_yolo_conf)
    else:
        boxes = detect_line_boxes(region_bgr, min_line_h=min_line_height, merge_gap=merge_gap)
    if not boxes:
        h, w = region_bgr.shape[:2]
        boxes = detect_line_boxes(region_bgr, min_line_h=min_line_height, merge_gap=merge_gap)
        if not boxes:
            boxes = [(0, 0, w - 1, h - 1)]

    lines_dir = output_dir / "lines"
    context_dir = output_dir / "context"
    lines_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)

    context = region_bgr.copy()
    crops: list[LineCrop] = []
    h, w = region_bgr.shape[:2]
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        pad = 4
        x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
        x2p, y2p = min(w - 1, x2 + pad), min(h - 1, y2 + pad)
        if x2p - x1p < 30 or y2p - y1p < 8:
            continue
        line_id = f"{region.region_id}_l{idx:02d}"
        line_path = lines_dir / f"{line_id}.png"
        cv2.imwrite(str(line_path), region_bgr[y1p:y2p, x1p:x2p])
        cv2.rectangle(context, (x1p, y1p), (x2p, y2p), (0, 220, 0), 2)
        crops.append(
            LineCrop(
                region.page_id,
                region.region_id,
                line_id,
                line_path,
                context_dir / f"{region.region_id}_context.png",
                region.x1 + x1p,
                region.y1 + y1p,
                region.x1 + x2p,
                region.y1 + y2p,
            )
        )
    cv2.imwrite(str(context_dir / f"{region.region_id}_context.png"), context)
    return crops


def create_word_crops(
    line: LineCrop,
    output_dir: Path,
    min_word_width: int,
    min_word_height: int,
    word_merge_gap: int,
) -> list[dict]:
    line_bgr = cv2.imread(str(line.image_path))
    if line_bgr is None:
        return []

    boxes = detect_word_boxes(
        line_bgr,
        min_word_width=min_word_width,
        min_word_height=min_word_height,
        merge_gap=word_merge_gap,
    )
    if not boxes:
        return []

    words_dir = output_dir / "words"
    context_dir = output_dir / "word_context"
    words_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)

    context = line_bgr.copy()
    h, w = line_bgr.shape[:2]
    rows = []
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        pad = 3
        x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
        x2p, y2p = min(w - 1, x2 + pad), min(h - 1, y2 + pad)
        if x2p - x1p < min_word_width or y2p - y1p < min_word_height:
            continue

        word_id = f"{line.line_id}_w{idx:02d}"
        word_path = words_dir / f"{word_id}.png"
        context_path = context_dir / f"{line.line_id}_word_context.png"
        cv2.imwrite(str(word_path), line_bgr[y1p:y2p, x1p:x2p])
        cv2.rectangle(context, (x1p, y1p), (x2p, y2p), (0, 120, 255), 2)
        rows.append(
            {
                "page_id": line.page_id,
                "region_id": line.region_id,
                "line_id": line.line_id,
                "word_id": word_id,
                "word_image_path": str(word_path),
                "line_image_path": str(line.image_path),
                "line_context_image_path": str(context_path),
                "x1_page": line.x1_page + x1p,
                "y1_page": line.y1_page + y1p,
                "x2_page": line.x1_page + x2p,
                "y2_page": line.y1_page + y2p,
            }
        )
    cv2.imwrite(str(context_dir / f"{line.line_id}_word_context.png"), context)
    return rows


def build_ocr_backend(args: argparse.Namespace) -> OCRBackend:
    if args.demo_texts:
        texts = [normalize_text(x) for x in args.demo_texts.read_text(encoding="utf-8").splitlines() if normalize_text(x)]
        return DemoTextOCR(texts)
    if args.ocr_backend == "none":
        return EmptyOCR()
    return TrOCROCR(args.trocr_model, args.max_target_len, args.num_beams)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full prescription recognition pipeline.")
    parser.add_argument("--input", type=Path, required=True, help="Input image or folder of prescription images.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/final_demo"), help="Output folder.")
    parser.add_argument("--labels-dir", type=Path, default=None, help="Optional YOLO layout labels folder.")
    parser.add_argument("--yolo-model", type=Path, default=None, help="Optional trained YOLO .pt model for handwritten-region detection.")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="Confidence threshold for --yolo-model predictions.")
    parser.add_argument("--target-class", type=int, default=1, help="YOLO class id for handwritten_region.")
    parser.add_argument("--line-yolo-model", type=Path, default=None, help="Optional trained YOLO .pt model for line detection inside regions.")
    parser.add_argument("--line-yolo-conf", type=float, default=0.25, help="Confidence threshold for --line-yolo-model predictions.")
    parser.add_argument("--line-target-class", type=int, default=0, help="YOLO class id for handwritten_line.")
    parser.add_argument("--use-full-page", action="store_true", help="Skip region proposal and segment the full page.")
    parser.add_argument("--max-side", type=int, default=2200, help="Max page dimension after preprocessing.")
    parser.add_argument("--min-line-height", type=int, default=14)
    parser.add_argument("--merge-gap", type=int, default=8)
    parser.add_argument("--ocr-backend", choices=["trocr", "none"], default="trocr")
    parser.add_argument("--ocr-unit", choices=["line", "word"], default="word", help="Run OCR on line crops or word crops.")
    parser.add_argument("--trocr-model", default="microsoft/trocr-base-handwritten")
    parser.add_argument("--max-target-len", type=int, default=48)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--demo-texts", type=Path, default=None, help="One OCR text line per detected line crop.")
    parser.add_argument("--min-word-width", type=int, default=16)
    parser.add_argument("--min-word-height", type=int, default=8)
    parser.add_argument("--word-merge-gap", type=int, default=8)
    parser.add_argument("--lexicon", type=Path, default=Path("pipeline/config/drug_lexicon.txt"))
    parser.add_argument("--match-threshold", type=float, default=0.72)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = args.output_dir / "pages"
    regions_dir = args.output_dir / "regions"
    lines_root = args.output_dir / "line_crops"
    for path in [pages_dir, regions_dir, lines_root]:
        path.mkdir(parents=True, exist_ok=True)

    lexicon = load_lexicon(args.lexicon)
    ocr = build_ocr_backend(args)
    yolo_model = None
    line_yolo_model = None
    if args.yolo_model is not None:
        from ultralytics import YOLO

        yolo_model = YOLO(str(args.yolo_model))
    if args.line_yolo_model is not None:
        from ultralytics import YOLO

        line_yolo_model = YOLO(str(args.line_yolo_model))

    page_rows: list[dict] = []
    region_rows: list[dict] = []
    line_rows: list[dict] = []
    word_rows: list[dict] = []
    prediction_rows: list[dict] = []

    for src_path in list_images(args.input):
        raw = cv2.imread(str(src_path))
        if raw is None:
            continue
        processed, angle = preprocess_page(raw, args.max_side)
        page_id = safe_id(src_path)
        page_path = pages_dir / f"{page_id}.jpg"
        cv2.imwrite(str(page_path), processed, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        page_rows.append(
            {
                "page_id": page_id,
                "source_path": str(src_path),
                "processed_path": str(page_path),
                "deskew_angle_deg": round(angle, 3),
                "width": processed.shape[1],
                "height": processed.shape[0],
            }
        )

        regions = create_regions(
            page_path,
            processed,
            regions_dir,
            args.labels_dir,
            args.target_class,
            args.use_full_page,
            yolo_model=yolo_model,
            yolo_conf=args.yolo_conf,
        )
        for region in regions:
            region_rows.append(
                {
                    "page_id": region.page_id,
                    "region_id": region.region_id,
                    "region_image_path": str(region.image_path),
                    "x1_page": region.x1,
                    "y1_page": region.y1,
                    "x2_page": region.x2,
                    "y2_page": region.y2,
                }
            )
            lines = create_line_crops(
                region,
                lines_root,
                args.min_line_height,
                args.merge_gap,
                line_yolo_model=line_yolo_model,
                line_target_class=args.line_target_class,
                line_yolo_conf=args.line_yolo_conf,
            )
            for line in lines:
                line_row = {
                    "page_id": line.page_id,
                    "region_id": line.region_id,
                    "line_id": line.line_id,
                    "line_image_path": str(line.image_path),
                    "region_image_path": str(region.image_path),
                    "context_image_path": str(line.context_path),
                    "x1_region": line.x1_page - region.x1,
                    "y1_region": line.y1_page - region.y1,
                    "x2_region": line.x2_page - region.x1,
                    "y2_region": line.y2_page - region.y1,
                    "x1_page": line.x1_page,
                    "y1_page": line.y1_page,
                    "x2_page": line.x2_page,
                    "y2_page": line.y2_page,
                }
                line_rows.append(line_row)

                if args.ocr_unit == "line":
                    ocr_text = ocr.predict(line.image_path)
                    parsed = parse_medical_entities(ocr_text, lexicon, args.match_threshold)
                    prediction_rows.append(
                        {
                            "ocr_unit": "line",
                            "page_id": line.page_id,
                            "region_id": line.region_id,
                            "line_id": line.line_id,
                            "word_id": "",
                            "image_path": str(line.image_path),
                            **parsed,
                        }
                    )
                    continue

                words = create_word_crops(
                    line,
                    lines_root,
                    min_word_width=args.min_word_width,
                    min_word_height=args.min_word_height,
                    word_merge_gap=args.word_merge_gap,
                )
                word_rows.extend(words)
                for word in words:
                    image_path = Path(str(word["word_image_path"]))
                    ocr_text = ocr.predict(image_path)
                    parsed = parse_medical_entities(ocr_text, lexicon, args.match_threshold)
                    prediction_rows.append(
                        {
                            "ocr_unit": "word",
                            "page_id": word["page_id"],
                            "region_id": word["region_id"],
                            "line_id": word["line_id"],
                            "word_id": word["word_id"],
                            "image_path": str(image_path),
                            **parsed,
                        }
                    )

    write_csv(args.output_dir / "page_manifest.csv", page_rows, ["page_id", "source_path", "processed_path", "deskew_angle_deg", "width", "height"])
    write_csv(args.output_dir / "region_manifest.csv", region_rows, ["page_id", "region_id", "region_image_path", "x1_page", "y1_page", "x2_page", "y2_page"])
    write_csv(args.output_dir / "line_manifest.csv", line_rows, ["page_id", "region_id", "line_id", "line_image_path", "region_image_path", "context_image_path", "x1_region", "y1_region", "x2_region", "y2_region", "x1_page", "y1_page", "x2_page", "y2_page"])
    write_csv(args.output_dir / "word_manifest.csv", word_rows, ["page_id", "region_id", "line_id", "word_id", "word_image_path", "line_image_path", "line_context_image_path", "x1_page", "y1_page", "x2_page", "y2_page"])
    write_csv(
        args.output_dir / "predictions.csv",
        prediction_rows,
        ["ocr_unit", "page_id", "region_id", "line_id", "word_id", "image_path", "ocr_text", "medicine_name", "medicine_match_score", "dosage", "frequency", "validation_status"],
    )
    (args.output_dir / "predictions.json").write_text(json.dumps(prediction_rows, indent=2), encoding="utf-8")

    print("End-to-end run complete.")
    print(f"Pages: {len(page_rows)}")
    print(f"Regions: {len(region_rows)}")
    print(f"Lines: {len(line_rows)}")
    print(f"Words: {len(word_rows)}")
    print(f"Predictions: {args.output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
