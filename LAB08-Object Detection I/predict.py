"""Detect cars in Kaggle testing images and save annotated images and submission CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=ROOT / "runs" / "yolo11s" / "weights" / "best.pt")
    parser.add_argument("--images", type=Path, default=ROOT / "data" / "testing_images")
    parser.add_argument("--output", type=Path, default=ROOT / "predictions")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(f"Model weights not found: {args.model}. Run train.py first.")
    if not args.images.is_dir():
        raise FileNotFoundError(f"Test images folder not found: {args.images}")

    image_files = sorted(p for p in args.images.iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not image_files:
        raise FileNotFoundError(f"No images found in {args.images}")

    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    rows = []
    for image_path in image_files:
        result = model.predict(source=str(image_path), conf=args.conf,
                               device=args.device, verbose=False)[0]
        with Image.open(image_path) as image:
            width, height = image.size
        values = []
        if result.boxes is not None:
            for xyxy in result.boxes.xyxy.cpu().tolist():
                x1, y1, x2, y2 = xyxy
                values.extend([
                    f"{max(0.0, min(x1 / width, 1.0)):.6f}",
                    f"{max(0.0, min(y1 / height, 1.0)):.6f}",
                    f"{max(0.0, min(x2 / width, 1.0)):.6f}",
                    f"{max(0.0, min(y2 / height, 1.0)):.6f}",
                ])
        rows.append({"image": image_path.name, "bounds": " ".join(values)})
        result.save(filename=str(args.output / image_path.name))

    submission_path = args.output / "submission.csv"
    with submission_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "bounds"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Processed {len(rows)} images")
    print(f"Annotated images and submission: {args.output}")


if __name__ == "__main__":
    main()
