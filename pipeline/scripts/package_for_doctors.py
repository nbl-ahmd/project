#!/usr/bin/env python3
"""
Create per-doctor annotation package with images + CSV.
Useful when sending tasks to external annotators.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package annotation tasks for one doctor.")
    parser.add_argument("--annotations-csv", type=Path, required=True, help="Master/per-doctor annotation CSV")
    parser.add_argument(
        "--annotator-id",
        type=str,
        default="",
        help="If set, export rows assigned to this annotator_id only.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Package output directory")
    return parser.parse_args()


def copy_if_exists(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lines_dir = args.output_dir / "line_images"
    ctx_dir = args.output_dir / "context_images"
    lines_dir.mkdir(parents=True, exist_ok=True)
    ctx_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.annotations_csv)
    if args.annotator_id:
        df = df[df["annotator_id"].astype(str) == args.annotator_id].copy()

    if len(df) == 0:
        raise ValueError("No rows to export after filtering.")

    out_rows = []
    base_dir = args.annotations_csv.parent

    for _, row in df.iterrows():
        line_src = Path(str(row["line_image_path"]))
        ctx_src = Path(str(row["context_image_path"]))
        if not line_src.is_absolute():
            line_src = (base_dir / line_src).resolve()
        if not ctx_src.is_absolute():
            ctx_src = (base_dir / ctx_src).resolve()

        line_dst = lines_dir / line_src.name
        ctx_dst = ctx_dir / ctx_src.name
        if not line_src.exists():
            continue
        copy_if_exists(line_src, line_dst)
        copy_if_exists(ctx_src, ctx_dst)

        row_out = row.copy()
        row_out["line_image_path"] = str(Path("line_images") / line_dst.name)
        row_out["context_image_path"] = str(Path("context_images") / ctx_dst.name)
        out_rows.append(row_out)

    out_df = pd.DataFrame(out_rows)
    out_csv = args.output_dir / "annotations.csv"
    out_df.to_csv(out_csv, index=False)

    print(f"Packaged rows: {len(out_df)}")
    print(f"Output: {args.output_dir}")
    print(f"CSV: {out_csv}")


if __name__ == "__main__":
    main()
