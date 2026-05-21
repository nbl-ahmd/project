#!/usr/bin/env python3
"""
Prepare processed page images for layout annotation in CVAT/Label Studio.

Creates:
- images folder (copied or symlinked)
- classes.txt
- manifest.csv
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd


VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> Iterable[Path]:
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            yield p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create package for layout annotation.")
    parser.add_argument("--pages-dir", type=Path, required=True, help="Processed page images.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output annotation package dir.")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["header", "handwritten_region", "footer"],
        help="Class names in index order.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        default=True,
        help="Copy images into the package. This is the default and is required on Google Drive.",
    )
    parser.add_argument(
        "--symlink-images",
        action="store_true",
        help="Use symlinks instead of copies. Do not use this on Google Drive/Colab.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_out = args.output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    rows = []
    for src in list_images(args.pages_dir):
        dst = images_out / src.name
        if dst.exists():
            dst.unlink()
        if args.symlink_images:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)

        rows.append({"image_name": src.name, "image_path": str(dst)})

    classes_path = args.output_dir / "classes.txt"
    classes_path.write_text("\n".join(args.classes) + "\n", encoding="utf-8")

    manifest_path = args.output_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    print(f"Prepared images: {len(rows)}")
    print(f"Classes file: {classes_path}")
    print(f"Manifest: {manifest_path}")
    print(
        "Next: annotate these images in CVAT/Label Studio and export YOLO labels with matching class order."
    )


if __name__ == "__main__":
    main()
