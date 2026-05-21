#!/usr/bin/env python3
"""
Combine multiple handwritten prescription word datasets into one TrOCR dataset.

Output layout:
output_root/
  Training/training_words/*
  Training/training_labels.csv
  Validation/validation_words/*
  Validation/validation_labels.csv
  Testing/testing_words/*
  Testing/testing_labels.csv

The output CSVs use the columns expected by the TrOCR notebook:
IMAGE, MEDICINE_NAME, SOURCE_DATASET, SOURCE_IMAGE
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def normalize_label(value: str) -> str:
    return " ".join(str(value).strip().split())


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._") or "sample"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def add_sample(
    samples: list[dict[str, str]],
    source_dataset: str,
    image_path: Path,
    label: str,
) -> None:
    label = normalize_label(label)
    if not label:
        return
    if not image_path.exists() or image_path.suffix.lower() not in VALID_EXTS:
        return
    samples.append(
        {
            "source_dataset": source_dataset,
            "image_path": str(image_path),
            "label": label,
            "source_image": image_path.name,
        }
    )


def collect_bd_dataset(root: Path) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    split_specs = [
        ("Training", "training_words", "training_labels.csv"),
        ("Validation", "validation_words", "validation_labels.csv"),
        ("Testing", "testing_words", "testing_labels.csv"),
    ]
    for split, image_dir_name, csv_name in split_specs:
        csv_path = root / split / csv_name
        image_dir = root / split / image_dir_name
        if not csv_path.exists() or not image_dir.exists():
            continue
        for row in read_csv_rows(csv_path):
            add_sample(
                samples,
                f"bd_{split.lower()}",
                image_dir / str(row.get("IMAGE", "")),
                row.get("MEDICINE_NAME", ""),
            )
    return samples


def collect_rxhand_dataset(root: Path) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    csv_path = root / "Prescription_Labels.csv"
    image_dir = root / "RxHand-Handwritten Prescription Word Image Dataset"
    if not csv_path.exists() or not image_dir.exists():
        return samples
    for row in read_csv_rows(csv_path):
        add_sample(samples, "rxhand", image_dir / str(row.get("Images", "")), row.get("Text", ""))
    return samples


def collect_archive_dataset(root: Path) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    csv_path = root / "doctor_handwriting_labels.csv"
    image_dir = root / "img" / "img"
    if not csv_path.exists() or not image_dir.exists():
        return samples
    for row in read_csv_rows(csv_path):
        add_sample(samples, "archive_doctor_handwriting", image_dir / str(row.get("filename", "")), row.get("label", ""))
    return samples


def stratified_split(
    samples: list[dict[str, str]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, str]]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for sample in samples:
        by_label[sample["label"].lower()].append(sample)

    splits = {"train": [], "val": [], "test": []}
    for label_samples in by_label.values():
        rng.shuffle(label_samples)
        n = len(label_samples)
        if n == 1:
            splits["train"].extend(label_samples)
            continue
        n_train = max(1, int(round(n * train_ratio)))
        n_val = int(round(n * val_ratio))
        if n >= 3 and n_val == 0:
            n_val = 1
        if n_train + n_val >= n and n >= 3:
            n_train = max(1, n - n_val - 1)
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n - n_train

        splits["train"].extend(label_samples[:n_train])
        splits["val"].extend(label_samples[n_train : n_train + n_val])
        splits["test"].extend(label_samples[n_train + n_val :])

    for rows in splits.values():
        rng.shuffle(rows)
    return splits


def write_split(output_root: Path, split_name: str, rows: list[dict[str, str]]) -> None:
    split_to_folder = {
        "train": ("Training", "training_words", "training_labels.csv"),
        "val": ("Validation", "validation_words", "validation_labels.csv"),
        "test": ("Testing", "testing_words", "testing_labels.csv"),
    }
    split_dir, image_dir_name, csv_name = split_to_folder[split_name]
    image_dir = output_root / split_dir / image_dir_name
    image_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / split_dir / csv_name

    label_rows = []
    for idx, row in enumerate(rows):
        src = Path(row["image_path"])
        out_name = f"{idx:06d}_{safe_name(row['source_dataset'])}{src.suffix.lower()}"
        dst = image_dir / out_name
        shutil.copy2(src, dst)
        label_rows.append(
            {
                "IMAGE": out_name,
                "MEDICINE_NAME": row["label"],
                "SOURCE_DATASET": row["source_dataset"],
                "SOURCE_IMAGE": row["source_image"],
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["IMAGE", "MEDICINE_NAME", "SOURCE_DATASET", "SOURCE_IMAGE"],
        )
        writer.writeheader()
        writer.writerows(label_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine TrOCR word datasets.")
    parser.add_argument(
        "--bd-root",
        type=Path,
        default=Path("data/doctors-handwritten-prescription-bd-dataset/Doctor’s Handwritten Prescription BD dataset"),
    )
    parser.add_argument(
        "--rxhand-root",
        type=Path,
        default=Path("data/new dataset/RxHand Original"),
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("data/new dataset/archive (1)"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/combined_trocr_word_dataset"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists. Pass --overwrite to rebuild it.")
        shutil.rmtree(args.output_root)

    samples = []
    sources = [
        ("BD", collect_bd_dataset(args.bd_root)),
        ("RxHand", collect_rxhand_dataset(args.rxhand_root)),
        ("Archive", collect_archive_dataset(args.archive_root)),
    ]
    for name, rows in sources:
        samples.extend(rows)
        print(f"{name}: {len(rows)} samples")

    if not samples:
        raise ValueError("No samples found.")

    splits = stratified_split(samples, args.train_ratio, args.val_ratio, args.seed)
    for split_name, rows in splits.items():
        write_split(args.output_root, split_name, rows)

    print("Combined TrOCR dataset created.")
    print(f"Output root: {args.output_root}")
    print(f"Train: {len(splits['train'])}")
    print(f"Validation: {len(splits['val'])}")
    print(f"Testing: {len(splits['test'])}")
    print(f"Total: {sum(len(v) for v in splits.values())}")


if __name__ == "__main__":
    main()
