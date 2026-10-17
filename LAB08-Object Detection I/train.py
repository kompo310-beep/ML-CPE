"""Train a pretrained Ultralytics YOLO detector on the prepared car dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "dataset" / "data.yaml")
    parser.add_argument("--model", default="yolo11s.pt",
                        help="Ultralytics model checkpoint, e.g. yolo11n.pt or yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="Training device (default: cpu)")
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset config not found: {args.data}. Run prepare_dataset.py first.")

    model = YOLO(args.model)  # pretrained weights are downloaded automatically on first run
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,  # safe default for Windows
        project=str(ROOT / "runs"),
        name="yolo11s",
        pretrained=True,
        plots=True,
        seed=42,
    )
    print(f"Training complete. Best weights: {ROOT / 'runs' / 'yolo11s' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
