#!/usr/bin/env python3
"""Train a YOLO layout detector for prescription handwritten-region boxes."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO handwritten-region detector.")
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", type=Path, default=Path("runs/layout"))
    parser.add_argument("--name", type=str, default="handwritten_region_yolo")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--degrees", type=float, default=5.0, help="Rotation augmentation for skewed prescription pages.")
    parser.add_argument("--shear", type=float, default=1.5, help="Shear augmentation for camera-captured pages.")
    parser.add_argument("--perspective", type=float, default=0.0005, help="Perspective augmentation for page photographs.")
    parser.add_argument("--scale", type=float, default=0.35, help="Scale augmentation.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
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
        "patience": args.patience,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    results = model.train(**train_kwargs)
    print("Training complete.")
    print(results)
    print("Best weights should be under:")
    print(args.project / args.name / "weights" / "best.pt")


if __name__ == "__main__":
    main()
