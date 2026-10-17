"""Convert the Kaggle car bounding-box CSV to Ultralytics YOLO format."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help="Extracted Kaggle data folder containing training_images and CSV")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.05 <= args.val_ratio <= 0.5:
        raise ValueError("--val-ratio must be between 0.05 and 0.5")

    images_dir = args.data_dir / "training_images"
    csv_path = args.data_dir / "train_solution_bounding_boxes (1).csv"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Training images folder not found: {images_dir}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Bounding-box CSV not found: {csv_path}")

    boxes_by_image: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"image", "xmin", "ymin", "xmax", "ymax"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must have columns: {', '.join(sorted(required))}")
        for row in reader:
            boxes_by_image[row["image"]].append(tuple(float(row[k]) for k in
                ("xmin", "ymin", "xmax", "ymax")))

    # A chronological split is useful for video frames: later frames are kept for validation.
    filenames = sorted(boxes_by_image, key=lambda name: _frame_number(name))
    if len(filenames) < 2:
        raise ValueError("At least two labeled images are required for train/validation split")
    val_count = max(1, round(len(filenames) * args.val_ratio))
    split = {name: ("val" if i >= len(filenames) - val_count else "train")
             for i, name in enumerate(filenames)}

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    written_images = 0
    written_boxes = 0
    missing = []
    for filename, boxes in boxes_by_image.items():
        source = images_dir / filename
        if not source.is_file():
            missing.append(filename)
            continue
        subset = split[filename]
        image_out = args.output_dir / "images" / subset / filename
        label_out = args.output_dir / "labels" / subset / (Path(filename).stem + ".txt")
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, image_out)

        with Image.open(source) as image:
            width, height = image.size
        if width <= 0 or height <= 0:
            continue

        lines = []
        for x1, y1, x2, y2 in boxes:
            x1 = max(0.0, min(x1, width))
            x2 = max(0.0, min(x2, width))
            y1 = max(0.0, min(y1, height))
            y2 = max(0.0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                continue
            xc = ((x1 + x2) / 2) / width
            yc = ((y1 + y2) / 2) / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        label_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written_images += 1
        written_boxes += len(lines)

    dataset_yaml = {
        "path": str(args.output_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "car"},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    print(f"Prepared {written_images} images and {written_boxes} car boxes")
    print(f"Train: {sum(v == 'train' for v in split.values())}; validation: {val_count}")
    print(f"YOLO dataset config: {args.output_dir / 'data.yaml'}")
    if missing:
        print(f"Warning: skipped {len(missing)} images missing from {images_dir}")


def _frame_number(filename: str) -> tuple[int, str]:
    """Sort vid_4_123.jpg by its numeric frame suffix, with a stable fallback."""
    try:
        return int(Path(filename).stem.rsplit("_", 1)[1]), filename
    except (IndexError, ValueError):
        return (0, filename)


if __name__ == "__main__":
    main()
