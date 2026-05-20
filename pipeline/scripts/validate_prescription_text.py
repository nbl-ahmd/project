#!/usr/bin/env python3
"""Validate OCR text lines against a drug lexicon and dosage/frequency rules."""

from __future__ import annotations

import argparse
import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path


DOSAGE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|gm|ml|iu|unit|units|tab|tabs|tablet|cap|caps|drop|drops|sachet)s?\b",
    re.IGNORECASE,
)
FREQUENCY_RE = re.compile(
    r"\b(?:\d\s*-\s*\d\s*-\s*\d|od|bd|tds|qid|hs|sos|prn|daily|night|morning|evening|weekly|once|twice)\b",
    re.IGNORECASE,
)


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().split())


def load_lexicon(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Lexicon not found: {path}")
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            key = next((c for c in ["medicine_name", "MEDICINE_NAME", "drug", "name"] if c in (reader.fieldnames or [])), None)
            if not key:
                raise ValueError(f"No supported medicine column found in {path}")
            values = [normalize_text(row.get(key, "")) for row in reader]
    else:
        values = [normalize_text(x) for x in path.read_text(encoding="utf-8").splitlines()]
    return sorted({x for x in values if x})


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def best_match(text: str, lexicon: list[str]) -> tuple[str, float]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]*", text)
    candidates = [text] + tokens
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


def parse_line(text: str, lexicon: list[str], threshold: float) -> dict[str, str | float]:
    text = normalize_text(text)
    drug, score = best_match(text, lexicon)
    dosages = [normalize_text(x) for x in DOSAGE_RE.findall(text)]
    frequencies = [normalize_text(x).upper() for x in FREQUENCY_RE.findall(text)]
    return {
        "ocr_text": text,
        "medicine_name": drug if score >= threshold else "",
        "medicine_match_score": score,
        "dosage": "; ".join(dict.fromkeys(dosages)),
        "frequency": "; ".join(dict.fromkeys(frequencies)),
        "validation_status": "matched" if score >= threshold else "needs_review",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate OCR text against drug lexicon.")
    parser.add_argument("--text", type=str, default="", help="Single OCR text line.")
    parser.add_argument("--text-file", type=Path, default=None, help="File with one OCR text line per row.")
    parser.add_argument("--lexicon", type=Path, default=Path("pipeline/config/drug_lexicon.txt"))
    parser.add_argument("--match-threshold", type=float, default=0.72)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.text and not args.text_file:
        raise ValueError("Provide --text or --text-file.")

    lexicon = load_lexicon(args.lexicon)
    lines = [args.text] if args.text else args.text_file.read_text(encoding="utf-8").splitlines()
    rows = [parse_line(line, lexicon, args.match_threshold) for line in lines if normalize_text(line)]

    payload = json.dumps(rows, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
