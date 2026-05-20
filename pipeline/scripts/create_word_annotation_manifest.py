#!/usr/bin/env python3
"""Create a word-level annotation CSV from word_manifest.csv."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create word annotation template CSV.")
    parser.add_argument("--word-manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--split-into-annotators", type=int, default=0)
    parser.add_argument("--annotator-prefix", type=str, default="annotator")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    words = pd.read_csv(args.word_manifest)
    required = {
        "word_id",
        "line_id",
        "page_id",
        "word_image_path",
        "line_context_image_path",
        "line_image_path",
    }
    missing = required - set(words.columns)
    if missing:
        raise ValueError(f"word manifest missing columns: {sorted(missing)}")

    ann = words[
        [
            "word_id",
            "line_id",
            "region_id",
            "page_id",
            "word_image_path",
            "line_image_path",
            "line_context_image_path",
        ]
    ].copy()
    ann["word_text"] = ""
    ann["medicine_name"] = ""
    ann["is_medicine"] = ""
    ann["confidence"] = "medium"
    ann["annotator_id"] = ""
    ann["review_status"] = "pending"
    ann["notes"] = ""
    ann["updated_at"] = datetime.utcnow().isoformat(timespec="seconds")
    ann.to_csv(args.output_csv, index=False)
    print(f"Word annotation sheet: {args.output_csv} ({len(ann)} rows)")

    if args.split_into_annotators > 0:
        for i in range(args.split_into_annotators):
            shard = ann.iloc[i :: args.split_into_annotators].copy()
            shard["annotator_id"] = f"{args.annotator_prefix}_{i + 1}"
            out = args.output_csv.parent / f"{args.annotator_prefix}_{i + 1}_word_annotations.csv"
            shard.to_csv(out, index=False)
            print(f"Annotator sheet: {out} ({len(shard)} rows)")


if __name__ == "__main__":
    main()

