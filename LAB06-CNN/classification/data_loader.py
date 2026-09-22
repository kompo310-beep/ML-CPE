import os
import cv2
import numpy as np

from preprocessing import preprocess_image

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def find_dataset_root(data_path):
    """Find the folder that contains both train/ and test/.

    Kaggle zips are sometimes nested (chest_xray/chest_xray/train), so
    search downward instead of assuming a fixed layout.
    """

    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {os.path.abspath(data_path)}\n"
            "Download the Chest X-Ray dataset and extract it to "
            "chest_xray/ at the project root, or change DATA_PATH in main.py."
        )

    for root, dirs, _ in os.walk(data_path):
        if "train" in dirs and "test" in dirs:
            return root

    raise FileNotFoundError(
        f"No train/ and test/ folders found under {os.path.abspath(data_path)}"
    )


def detect_classes(split_path):
    """Class names = sub-folder names (NORMAL, PNEUMONIA), sorted."""

    return sorted(
        folder
        for folder in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, folder))
    )


def load_split(split_path, classes, img_size=150, max_per_class=None):
    """Load every class folder inside one split (train or test)."""

    images = []
    labels = []

    for label, class_name in enumerate(classes):
        class_path = os.path.join(split_path, class_name)

        if not os.path.isdir(class_path):
            raise FileNotFoundError(f"Missing class folder: {class_path}")

        filenames = sorted(
            f for f in os.listdir(class_path)
            if f.lower().endswith(VALID_EXT)
        )

        loaded = 0
        skipped = 0
        for filename in filenames:
            if max_per_class and loaded >= max_per_class:
                break

            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Resize here so full-size images are not all kept in memory
            image = preprocess_image(image, img_size)

            # Skip unreadable or damaged images
            if image is None:
                skipped += 1
                continue

            images.append(image)
            labels.append(label)
            loaded += 1

        print(f"  {class_name:<10}: {loaded} images ({skipped} skipped)")

    return np.stack(images), np.array(labels)


def load_data(data_path, img_size=150, max_per_class=None):
    """Returns X_train, y_train, X_test, y_test, classes."""

    root = find_dataset_root(data_path)
    train_dir = os.path.join(root, "train")
    test_dir = os.path.join(root, "test")

    # Use the same class order for both splits so labels always match
    classes = detect_classes(train_dir)
    print("Dataset root     :", os.path.abspath(root))
    print("Detected classes :", classes)

    print("\nLoading train split...")
    X_train, y_train = load_split(train_dir, classes, img_size, max_per_class)

    print("\nLoading test split...")
    X_test, y_test = load_split(test_dir, classes, img_size, max_per_class)

    return X_train, y_train, X_test, y_test, classes