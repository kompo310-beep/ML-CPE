import hashlib
import os

import cv2
import numpy as np

from preprocessing import preprocess_image

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")

# The Kaggle archive has several copies of the data. This is the one used
# when more than one candidate folder is found.
PREFERRED_DIR = "rps-cv-images"

# Folder names that mean "already split". If a dataset has all three, it is
# used as it is and never split again.
SPLIT_NAMES = {
    "train": ("train",),
    "val": ("val", "valid", "validation"),
    "test": ("test",),
}
RESERVED_NAMES = {name for names in SPLIT_NAMES.values() for name in names}

MAX_SEARCH_DEPTH = 3


def is_image(filename):
    # "._xxx.jpg" files are macOS metadata, not images
    return filename.lower().endswith(VALID_EXT) and not filename.startswith(".")


def list_class_dirs(path):
    """Sub-folders of `path` that directly contain images (= class folders)."""

    names = []
    for folder in sorted(os.listdir(path)):
        folder_path = os.path.join(path, folder)
        if folder.startswith((".", "__")) or not os.path.isdir(folder_path):
            continue
        if any(is_image(f) for f in os.listdir(folder_path)):
            names.append(folder)
    return names


def _walk_dirs(root, max_depth):
    root = os.path.abspath(root)
    base = root.count(os.sep)

    for current, dirs, _ in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith((".", "__")))
        if current.count(os.sep) - base >= max_depth:
            dirs[:] = []
        yield current


def _find_presplit(path):
    found = {}
    for key, aliases in SPLIT_NAMES.items():
        for alias in aliases:
            candidate = os.path.join(path, alias)
            if os.path.isdir(candidate) and len(list_class_dirs(candidate)) >= 2:
                found[key] = candidate
                break
    return found if len(found) == len(SPLIT_NAMES) else None


def find_dataset(data_path):
    """Locate the dataset below `data_path`.

    Returns {"mode": "presplit", "splits": {"train": ..., "val": ..., "test": ...}}
    or {"mode": "single", "root": <folder that contains the class folders>}.
    """

    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {os.path.abspath(data_path)}\n"
            "Download it from https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors\n"
            "and extract it there, or pass the folder with --data-path."
        )

    candidates = []
    for current in _walk_dirs(data_path, MAX_SEARCH_DEPTH):
        splits = _find_presplit(current)
        if splits:
            return {"mode": "presplit", "splits": splits}

        if os.path.basename(current).lower() in RESERVED_NAMES:
            continue
        if len(list_class_dirs(current)) >= 2:
            candidates.append(current)

    if not candidates:
        raise FileNotFoundError(
            f"No class folders (e.g. paper/, rock/, scissors/) with images "
            f"found under {os.path.abspath(data_path)}"
        )

    # Shallowest first; prefer the rps-cv-images folder of the Kaggle archive
    candidates.sort(key=lambda p: p.count(os.sep))
    preferred = [p for p in candidates
                 if os.path.basename(p).lower() == PREFERRED_DIR]
    root = (preferred or candidates)[0]

    if len(candidates) > 1:
        ignored = [p for p in candidates if p != root]
        print("Other class folders found and ignored:")
        for p in ignored:
            print("  ", p)

    return {"mode": "single", "root": root}


def _load_split(root, classes, img_size, max_per_class, seen_hashes):
    """Load every class folder of `root`; labels follow the order of `classes`."""

    images = []
    labels = []
    total_skipped = 0
    total_duplicates = 0

    for label, class_name in enumerate(classes):
        class_path = os.path.join(root, class_name)
        filenames = sorted(f for f in os.listdir(class_path) if is_image(f))

        loaded = 0
        skipped = 0
        duplicates = 0
        for filename in filenames:
            if max_per_class and loaded >= max_per_class:
                break

            # Read the bytes ourselves so the file can be hashed and decoded
            try:
                data = np.fromfile(os.path.join(class_path, filename),
                                   dtype=np.uint8)
            except OSError:
                skipped += 1
                continue

            if data.size == 0:
                skipped += 1
                continue

            # An identical file in two splits would leak into the test set
            digest = hashlib.md5(data.tobytes()).hexdigest()
            if digest in seen_hashes:
                duplicates += 1
                continue

            try:
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            except cv2.error:
                image = None

            # Resize here so full-size images are not all kept in memory
            image = preprocess_image(image, img_size)

            # Skip unreadable or damaged images
            if image is None:
                skipped += 1
                continue

            seen_hashes.add(digest)
            images.append(image)
            labels.append(label)
            loaded += 1

        print(f"Loaded class {class_name}: {loaded} images "
              f"({skipped} skipped, {duplicates} duplicates)")
        total_skipped += skipped
        total_duplicates += duplicates

    print(f"Skipped (unreadable or corrupted): {total_skipped}")
    print(f"Skipped (exact duplicate files)  : {total_duplicates}")

    if not images:
        raise ValueError(f"No readable images found in {os.path.abspath(root)}")

    return np.stack(images), np.array(labels), total_skipped


def load_data(root, img_size=100, max_per_class=None):
    """Load a dataset stored as root/<class_name>/<image files>."""

    # Detect classes automatically from subdirectories; the label of a class
    # is its position in this sorted list (paper=0, rock=1, scissors=2).
    classes = list_class_dirs(root)
    print("Detected classes:", classes)

    images, labels, _ = _load_split(root, classes, img_size,
                                    max_per_class, set())

    return images, labels, classes


def load_presplit(split_paths, img_size=100, max_per_class=None):
    """Load a dataset that already has train / val / test folders.

    Returns ({"train": (X, y), "val": (X, y), "test": (X, y)}, classes).
    The class list comes from the train folder and must match the others.
    """

    classes = list_class_dirs(split_paths["train"])
    print("Detected classes:", classes)

    for name, path in split_paths.items():
        if list_class_dirs(path) != classes:
            raise ValueError(
                f"Class folders in '{name}' ({list_class_dirs(path)}) "
                f"do not match 'train' ({classes})"
            )

    # One shared hash set: a file already used by train is dropped from val
    # and test, so nothing can appear in two splits.
    seen_hashes = set()
    splits = {}
    for name in ("train", "val", "test"):
        print(f"\n[{name}] {split_paths[name]}")
        X, y, _ = _load_split(split_paths[name], classes, img_size,
                              max_per_class, seen_hashes)
        splits[name] = (X, y)

    return splits, classes
