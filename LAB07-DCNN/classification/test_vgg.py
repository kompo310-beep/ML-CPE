import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import transfer_learning  # ลงทะเบียน VGG16 preprocessing ตอนโหลดโมเดล
from preprocessing import preprocess_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

N_SAMPLES = 4


def find_models(names=None):
    """Trained models are stored as outputs/<name>/<name>.keras"""

    found = {}
    if os.path.isdir(OUTPUT_DIR):
        for folder in sorted(os.listdir(OUTPUT_DIR)):
            path = os.path.join(OUTPUT_DIR, folder, f"{folder}.keras")
            if os.path.isfile(path):
                found[folder] = path

    if names:
        missing = [n for n in names if n not in found]
        if missing:
            raise FileNotFoundError(
                f"No trained model named {missing}. Available: {sorted(found)}"
            )
        found = {n: found[n] for n in names}

    if not found:
        raise FileNotFoundError("No trained model in outputs/. Run main.py first.")

    return found


def load_user_images(paths, img_size, classes):
    """Read image files; the true class is taken from the parent folder name
    when it is one of the classes (e.g. .../rock/0.png), otherwise None."""

    images = []
    true_names = []
    shown = []

    for path in paths:
        try:
            data = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        except (OSError, cv2.error):
            image = None

        image = preprocess_image(image, img_size)
        if image is None:
            print(f"Skipped (cannot read): {path}")
            continue

        parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
        images.append(image)
        true_names.append(parent if parent in classes else None)
        shown.append(os.path.basename(path))

    if not images:
        raise ValueError("None of the given images could be read")

    return np.stack(images), true_names, shown


def predict_and_plot(name, model, X_sample, true_names, names, classes):
    """Print and plot the predictions of one model.
    Returns (number correct, number with a known true class)."""

    print(f"\n--- {name} ---")

    probabilities = model.predict(X_sample, verbose=0)
    if probabilities.shape[1] != len(classes):
        raise ValueError(
            f"{name} has {probabilities.shape[1]} outputs but classes.json "
            f"lists {len(classes)} classes: re-run main.py"
        )

    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    n_samples = len(X_sample)
    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 4.0 * rows))
    axes = np.atleast_1d(axes).ravel()

    known = 0
    correct_total = 0

    for i, ax in enumerate(axes):
        if i >= n_samples:
            ax.axis("off")
            continue

        pred = classes[predictions[i]]
        true = true_names[i]

        if true is None:
            color = "black"
            true_line = "True: unknown"
            status = ""
        else:
            correct = pred == true
            known += 1
            correct_total += int(correct)
            color = "green" if correct else "red"
            true_line = f"True: {true.capitalize()}"
            status = "OK" if correct else "WRONG"

        ax.imshow(X_sample[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Pred: {pred.capitalize()} ({confidence[i] * 100:.0f}%)\n"
                     f"{true_line}", color=color)

        scores = "  ".join(f"{c} {p * 100:5.1f}%"
                           for c, p in zip(classes, probabilities[i]))
        print(f"[{i + 1}] {names[i]:<14} Pred: {pred:<9} "
              f"conf {confidence[i] * 100:5.1f}%  {status:<5} | {scores}")

    if known:
        print(f"Correct: {correct_total}/{known}")
        fig.suptitle(f"{name}: {correct_total}/{known} correct")
    else:
        fig.suptitle(name)
    fig.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, name, f"{name}_prediction_sample.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")

    return correct_total, known


def test_vgg(image_paths=None, n_samples=N_SAMPLES, model_names=None):

    models = find_models(model_names)

    with open(f"{OUTPUT_DIR}/classes.json") as f:
        classes = json.load(f)

    if not image_paths:
        X_test = np.load(f"{OUTPUT_DIR}/X_test.npy")
        y_test = np.load(f"{OUTPUT_DIR}/y_test.npy")

        # Pick random images once (no seed -> different every run); every
        # model is tested on these same images
        index = np.random.choice(len(X_test), min(n_samples, len(X_test)),
                                 replace=False)
        X_random = X_test[index]
        random_true = [classes[y_test[i]] for i in index]
        random_names = [f"test #{i}" for i in index]

    summary = []
    for name, path in models.items():
        model = keras.models.load_model(path)

        # The model knows its own input size (set by --img-size when training)
        img_size = model.input_shape[1]

        if image_paths:
            X_sample, true_names, names = load_user_images(
                image_paths, img_size, classes
            )
        elif X_random.shape[1] == img_size:
            X_sample, true_names, names = X_random, random_true, random_names
        else:
            print(f"\nSkipped {name}: it expects {img_size}px images but the "
                  f"saved test set is {X_random.shape[1]}px. "
                  f"Re-run main.py to train all models at the same size.")
            keras.backend.clear_session()
            continue

        summary.append((name,) + predict_and_plot(
            name, model, X_sample, true_names, names, classes
        ))
        keras.backend.clear_session()

    if len(summary) > 1:
        print("\n" + "=" * 30)
        for name, correct, known in summary:
            if known:
                print(f"{name:<16} {correct}/{known} correct")
        print("=" * 30)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Predict Rock / Paper / Scissors with the trained models"
    )
    parser.add_argument("images", nargs="*",
                        help="image files to classify (default: random "
                             "images from the saved test set)")
    parser.add_argument("-n", "--samples", type=int, default=N_SAMPLES,
                        help="number of random test images")
    parser.add_argument("--model", default=None,
                        help="model name, or several separated by commas, "
                             "e.g. vgg16,mobilenetv2 "
                             "(default: every trained model)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_names = args.model.split(",") if args.model else None
    test_vgg(args.images, args.samples, model_names)