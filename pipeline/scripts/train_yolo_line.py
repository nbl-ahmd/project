#!/usr/bin/env python3
"""Train a YOLO detector for handwritten line boxes inside prescription regions."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO handwritten-line detector.")
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", type=Path, default=Path("runs/lines"))
    parser.add_argument("--name", type=str, default="handwritten_line_yolo")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--degrees", type=float, default=8.0, help="Rotation augmentation for slanted handwritten lines.")
    parser.add_argument("--shear", type=float, default=2.0, help="Shear augmentation for angled handwriting.")
    parser.add_argument("--perspective", type=float, default=0.0005, help="Perspective augmentation for camera-captured pages.")
    parser.add_argument("--scale", type=float, default=0.35, help="Scale augmentation.")
    parser.add_argument("--mosaic", type=float, default=0.0, help="Mosaic augmentation. Keep 0 for thin line boxes.")
    parser.add_argument("--mixup", type=float, default=0.0, help="MixUp augmentation. Keep 0 for thin line boxes.")
    parser.add_argument("--copy-paste", type=float, default=0.0, help="Copy-paste augmentation. Keep 0 for line detection.")
    parser.add_argument("--close-mosaic", type=int, default=0, help="Disable mosaic from this many final epochs.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument(
        "--weights-out",
        type=Path,
        default=Path("models/line_yolo_best.pt"),
        help="Stable output path where best.pt is copied after training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(args.data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(args.project),
        "name": args.name,
        "exist_ok": True,
        "degrees": args.degrees,
        "shear": args.shear,
        "perspective": args.perspective,
        "scale": args.scale,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,
        "close_mosaic": args.close_mosaic,
        "patience": args.patience,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    results = model.train(**train_kwargs)
    print("Training complete.")
    print(results)
    run_dir = Path(getattr(results, "save_dir", args.project / args.name))
    best_path = run_dir / "weights" / "best.pt"
    if not best_path.exists():
        fallback = sorted(Path.cwd().glob(f"**/{args.name}/weights/best.pt"), key=lambda x: x.stat().st_mtime)
        if fallback:
            best_path = fallback[-1]
    if not best_path.exists():
        raise FileNotFoundError(f"Could not locate trained best.pt for run name: {args.name}")
    args.weights_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, args.weights_out)
    print("Best weights found at:")
    print(best_path)
    print("Stable weights copied to:")
    print(args.weights_out)


if __name__ == "__main__":
    main()
