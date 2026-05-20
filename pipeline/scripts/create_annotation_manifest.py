#!/usr/bin/env python3
"""
Create doctor annotation CSV template from line manifest.
Optionally split into per-doctor CSVs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create annotation template CSV from line manifest.")
    parser.add_argument("--line-manifest", type=Path, required=True, help="Input line_manifest.csv")
    parser.add_argument("--output-csv", type=Path, required=True, help="Master annotation CSV output path")
    parser.add_argument(
        "--split-into-doctors",
        type=int,
        default=0,
        help="If >0, create N per-doctor CSVs from the same rows.",
    )
    parser.add_argument(
        "--doctor-prefix",
        type=str,
        default="doctor",
        help="Per-doctor CSV prefix, e.g. doctor -> doctor_1_annotations.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    lines = pd.read_csv(args.line_manifest)
    required = {"line_id", "page_id", "line_image_path", "context_image_path"}
    missing = required - set(lines.columns)
    if missing:
        raise ValueError(f"line manifest missing columns: {sorted(missing)}")

    now = datetime.utcnow().isoformat(timespec="seconds")
    for optional_col in ["region_image_path", "page_image_path"]:
        if optional_col not in lines.columns:
            lines[optional_col] = ""

    ann = lines[
        [
            "line_id",
            "page_id",
            "line_image_path",
            "context_image_path",
            "region_image_path",
            "page_image_path",
        ]
    ].copy()
    ann["transcription"] = ""
    ann["medicine_name"] = ""
    ann["dosage"] = ""
    ann["frequency"] = ""
    ann["confidence"] = "medium"
    ann["annotator_id"] = ""
    ann["review_status"] = "pending"
    ann["notes"] = ""
    ann["updated_at"] = now

    ann.to_csv(args.output_csv, index=False)
    print(f"Master annotation sheet: {args.output_csv} ({len(ann)} rows)")

    if args.split_into_doctors > 0:
        n = args.split_into_doctors
        for i in range(n):
            shard = ann.iloc[i::n].copy()
            shard["annotator_id"] = f"{args.doctor_prefix}_{i + 1}"
            out = args.output_csv.parent / f"{args.doctor_prefix}_{i + 1}_annotations.csv"
            shard.to_csv(out, index=False)
            print(f"Per-doctor sheet: {out} ({len(shard)} rows)")


if __name__ == "__main__":
    main()
