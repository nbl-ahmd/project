#!/usr/bin/env python3
"""Validate OCR text lines against a drug lexicon and dosage/frequency rules."""

from __future__ import annotations

import argparse
import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path


NON_MEDICAL_TOKENS = {
    "rx",
    "symptom",
    "symptoms",
    "complaint",
    "complaints",
    "diagnosis",
    "advice",
    "adv",
    "test",
    "review",
    "follow",
    "followup",
    "follow-up",
    "patient",
    "name",
    "age",
    "sex",
    "date",
    "doctor",
    "dr",
    "signature",
    "clinic",
    "hospital",
}
DOSAGE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|gm|ml|iu|unit|units|tab|tabs|tablet|cap|caps|drop|drops|sachet|%)s?\b",
    re.IGNORECASE,
)
FREQUENCY_RE = re.compile(
    r"\b(?:\d\s*-\s*\d\s*-\s*\d|od|bd|bid|tds|tid|qid|qds|hs|sos|stat|prn|ac|pc|bbf|daily|night|morning|evening|weekly|once|twice|thrice)\b",
    re.IGNORECASE,
)
ROUTE_RE = re.compile(r"\b(?:po|oral|iv|im|sc|sl|topical|inh|nebulization|neb|drops?)\b", re.IGNORECASE)


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
    return sorted({x for x in values if x and x.lower() not in NON_MEDICAL_TOKENS})


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def clean_candidate(value: str) -> str:
    value = normalize_text(value)
    value = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", value)
    value = re.sub(r"\b(?:tab|tabs|tablet|cap|caps|capsule|syr|syp|inj|drop|drops|cream|oint|gel)\b", "", value, flags=re.IGNORECASE)
    value = DOSAGE_RE.sub("", value)
    value = FREQUENCY_RE.sub("", value)
    value = ROUTE_RE.sub("", value)
    return normalize_text(value)


def token_candidates(text: str) -> list[str]:
    cleaned = clean_candidate(text)
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]*", cleaned)
    candidates = [cleaned] if cleaned else []
    candidates.extend(tokens)
    if len(tokens) >= 2:
        candidates.extend(" ".join(tokens[i : i + 2]) for i in range(len(tokens) - 1))
    seen: set[str] = set()
    filtered: list[str] = []
    for candidate in candidates:
        candidate = clean_candidate(candidate)
        key = candidate.lower()
        if not candidate or key in seen or key in NON_MEDICAL_TOKENS or len(candidate) <= 1:
            continue
        seen.add(key)
        filtered.append(candidate)
    return filtered


def best_match(text: str, lexicon: list[str]) -> tuple[str, float, str]:
    best_name = ""
    best_score = 0.0
    best_candidate = ""
    for candidate in token_candidates(text):
        for drug in lexicon:
            candidate_l = candidate.lower()
            drug_l = drug.lower()
            score = similarity(candidate_l, drug_l)
            if candidate_l == drug_l:
                score = 1.0
            elif candidate_l in drug_l or drug_l in candidate_l:
                score = max(score, min(len(candidate_l), len(drug_l)) / max(len(candidate_l), len(drug_l)))
            if score > best_score:
                best_name = drug
                best_score = score
                best_candidate = candidate
    return best_name, round(best_score, 4), best_candidate


def parse_line(text: str, lexicon: list[str], threshold: float) -> dict[str, str | float]:
    text = normalize_text(text)
    drug, score, matched_candidate = best_match(text, lexicon)
    dosages = [normalize_text(x) for x in DOSAGE_RE.findall(text)]
    frequencies = [normalize_text(x).upper() for x in FREQUENCY_RE.findall(text)]
    routes = [normalize_text(x).upper() for x in ROUTE_RE.findall(text)]
    is_medicine = bool(drug and score >= threshold)
    has_dose_info = bool(dosages or frequencies or routes)
    return {
        "ocr_text": text,
        "medicine_name": drug if is_medicine else "",
        "medicine_match_score": score,
        "matched_candidate": matched_candidate,
        "dosage": "; ".join(dict.fromkeys(dosages)),
        "frequency": "; ".join(dict.fromkeys(frequencies)),
        "route": "; ".join(dict.fromkeys(routes)),
        "keep_for_output": "yes" if is_medicine or has_dose_info else "no",
        "validation_status": "matched" if is_medicine else ("dose_only" if has_dose_info else "ignored_non_medicine"),
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
