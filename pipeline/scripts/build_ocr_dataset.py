#!/usr/bin/env python3
"""
Build OCR dataset splits from reviewed annotation CSV.

Creates:
output_root/
  Training/training_words/*.png
  Validation/validation_words/*.png
  Testing/testing_words/*.png
  Training/training_labels.csv
  Validation/validation_labels.csv
  Testing/testing_labels.csv
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OCR dataset splits from annotation CSV.")
    parser.add_argument("--annotations-csv", type=Path, required=True, help="Doctor annotation CSV")
    parser.add_argument("--output-root", type=Path, required=True, help="Output dataset root")
    parser.add_argument("--label-column", type=str, default="medicine_name", help="Target label column")
    parser.add_argument(
        "--image-path-column",
        type=str,
        default="line_image_path",
        help="Image path column to export. Use word_image_path for word-level OCR.",
    )
    parser.add_argument(
        "--approved-status",
        type=str,
        default="reviewed",
        help="Comma-separated review_status values allowed for export",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_label(x: str) -> str:
    return " ".join(str(x).strip().split())


def split_counts(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if n <= 1:
        return n, 0, 0

    n_train = max(1, int(round(n * train_ratio)))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    if n_test < 0:
        n_val = max(0, n_val + n_test)
        n_test = 0
    if n_test == 0 and n >= 3:
        if n_train > 1:
            n_train -= 1
            n_test += 1
        elif n_val > 0:
            n_val -= 1
            n_test += 1
    if n_val == 0 and n >= 5:
        if n_train > 1:
            n_train -= 1
            n_val = 1
        elif n_test > 1:
            n_test -= 1
            n_val = 1

    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "train_words": root / "Training" / "training_words",
        "val_words": root / "Validation" / "validation_words",
        "test_words": root / "Testing" / "testing_words",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    df = pd.read_csv(args.annotations_csv)
    needed = {args.image_path_column, args.label_column}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"annotations CSV missing columns: {sorted(missing)}")

    if "review_status" in df.columns:
        approved = {x.strip() for x in args.approved_status.split(",") if x.strip()}
        df = df[df["review_status"].astype(str).isin(approved)].copy()

    df[args.label_column] = df[args.label_column].map(normalize_label)
    df = df[df[args.label_column] != ""].copy()
    df = df[df[args.image_path_column].astype(str).map(lambda p: Path(p).exists())].copy()

    if len(df) == 0:
        raise ValueError("No valid rows after filtering. Check review_status and label completeness.")

    split_rows = {"train": [], "val": [], "test": []}
    for _, group in df.groupby(args.label_column):
        idxs = list(group.index)
        rng.shuffle(idxs)
        n = len(idxs)
        n_train, n_val, n_test = split_counts(n, args.train_ratio, args.val_ratio)

        split_rows["train"].extend(idxs[:n_train])
        split_rows["val"].extend(idxs[n_train : n_train + n_val])
        split_rows["test"].extend(idxs[n_train + n_val : n_train + n_val + n_test])

    dirs = ensure_dirs(args.output_root)
    counters = {"train": 0, "val": 0, "test": 0}
    label_rows = {"train": [], "val": [], "test": []}

    split_to_dir = {"train": dirs["train_words"], "val": dirs["val_words"], "test": dirs["test_words"]}

    for split_name, idxs in split_rows.items():
        for idx in idxs:
            row = df.loc[idx]
            src = Path(row[args.image_path_column])
            out_name = f"{counters[split_name]}.png"
            dst = split_to_dir[split_name] / out_name
            shutil.copy2(src, dst)

            label_rows[split_name].append(
                {
                    "IMAGE": out_name,
                    "MEDICINE_NAME": row[args.label_column],
                }
            )
            counters[split_name] += 1

    pd.DataFrame(label_rows["train"]).to_csv(
        args.output_root / "Training" / "training_labels.csv", index=False
    )
    pd.DataFrame(label_rows["val"]).to_csv(
        args.output_root / "Validation" / "validation_labels.csv", index=False
    )
    pd.DataFrame(label_rows["test"]).to_csv(
        args.output_root / "Testing" / "testing_labels.csv", index=False
    )

    print("Dataset build complete.")
    print(f"Train: {len(label_rows['train'])}")
    print(f"Val: {len(label_rows['val'])}")
    print(f"Test: {len(label_rows['test'])}")
    print(f"Output root: {args.output_root}")


if __name__ == "__main__":
    main()
