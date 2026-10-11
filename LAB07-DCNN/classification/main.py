import argparse
import csv
import json
import os
import time
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

import efficientnet_model
import mobilenet_model
import resnet_model
import vgg_model
from data_loader import find_dataset, load_data, load_presplit
from evaluate import confusion_matrix, evaluate_model, plot_history
from preprocessing import to_features
from split_data import split_dataset, print_split_summary

# Paths are relative to this file, so the script runs from any directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Folder that holds the extracted Kaggle dataset. Change it here, set the
# RPS_DATA_PATH environment variable, or pass --data-path on the command line.
DATA_PATH = os.environ.get(
    "RPS_DATA_PATH", os.path.join(BASE_DIR, "..", "rockpaperscissors")
)

EXPECTED_CLASSES = {"paper", "rock", "scissors"}

IMG_SIZE = 96          # same size for every model, so the comparison is fair
TEST_SIZE = 0.2
VAL_SIZE = 0.1
MAX_PER_CLASS = None   # None = use all images
EPOCHS = 30
HEAD_EPOCHS = 5        # pretrained models: epochs with the backbone frozen
BATCH_SIZE = 32

SEED = 42




def _transfer(module):
    """Pretrained models, one file each (mobilenet_model.py, ...)."""

    def run(data, num_classes, epochs, batch_size, head_epochs):
        model, history = module.train_model(
            *data, num_classes, None, epochs, batch_size, head_epochs
        )
        return model, history.history

    return run


# name -> function(data, num_classes, epochs, batch_size, head_epochs)
# To add a model: write <name>_model.py (copy mobilenet_model.py) and add a line.
MODELS = {
    "vgg16": _transfer(vgg_model),
    "mobilenetv2": _transfer(mobilenet_model),
    "efficientnetb0": _transfer(efficientnet_model),
    "resnet50v2": _transfer(resnet_model),
}

DEFAULT_MODELS = ["vgg16", "mobilenetv2", "efficientnetb0"]


def parse_args():

    parser = argparse.ArgumentParser(
        description="Rock Paper Scissors: train several models and compare them"
    )
    parser.add_argument("--data-path", default=DATA_PATH,
                        help="folder with the extracted Kaggle dataset")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        choices=sorted(MODELS), help="models to train")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--head-epochs", type=int, default=HEAD_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-per-class", type=int, default=MAX_PER_CLASS,
                        help="limit images per class (default: all)")

    return parser.parse_args()


def load_splits(data_path, img_size, max_per_class):

    layout = find_dataset(data_path)

    if layout["mode"] == "presplit":
        print("Dataset already has train/val/test folders: using them as they are.")
        splits, classes = load_presplit(
            layout["splits"], img_size, max_per_class
        )
        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]
        X_test, y_test = splits["test"]
        X_train, X_val, X_test = (
            to_features(X_train), to_features(X_val), to_features(X_test)
        )
    else:
        print("Dataset folder:", layout["root"])
        images, labels, classes = load_data(
            layout["root"], img_size, max_per_class
        )
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
            to_features(images), labels, TEST_SIZE, VAL_SIZE, SEED
        )

    return X_train, X_val, X_test, y_train, y_val, y_test, classes


def macro_f1(matrix):

    tp = np.diag(matrix).astype(float)
    predicted = matrix.sum(axis=0)
    support = matrix.sum(axis=1)

    precision = np.divide(tp, predicted, out=np.zeros_like(tp),
                          where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp),
                       where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp),
                   where=denom > 0)

    return float(f1.mean())


def evaluate_one(name, model, history, train_time, X_test, y_test, classes):
    """Save everything of one model in outputs/<name>/ and return its numbers."""

    model_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{name}.keras")
    model.save(model_path)
    with open(os.path.join(model_dir, f"{name}_history.json"), "w") as f:
        json.dump({k: [float(v) for v in vs] for k, vs in history.items()}, f)
    print(f"Saved: {model_path}")

    # Warm-up call, then time the whole test set
    model.predict(X_test[:8], verbose=0)
    start = time.perf_counter()
    probabilities = model.predict(X_test, batch_size=32, verbose=0)
    ms_per_image = (time.perf_counter() - start) / len(X_test) * 1000

    predictions = probabilities.argmax(axis=1)

    accuracy = evaluate_model(
        y_test, predictions, classes,
        save_path=os.path.join(model_dir, f"{name}_confusion_matrix.png")
    )
    plot_history(SimpleNamespace(history=history),
                 os.path.join(model_dir, f"{name}_training_history.png"))

    matrix = confusion_matrix(y_test, predictions, len(classes))

    return {
        "model": name,
        "params_million": model.count_params() / 1e6,
        "size_mb": os.path.getsize(model_path) / 1e6,
        "epochs": len(history["loss"]),
        "train_time_min": train_time / 60,
        "test_accuracy": accuracy,
        "macro_f1": macro_f1(matrix),
        "ms_per_image": ms_per_image,
    }


def print_table(results):

    print("\n" + "=" * 92)
    print(f"{'model':<16}{'params(M)':>10}{'size(MB)':>10}{'epochs':>8}"
          f"{'time(min)':>11}{'test acc':>10}{'macro F1':>10}{'ms/img':>9}")
    print("-" * 92)
    for r in results:
        print(f"{r['model']:<16}{r['params_million']:>10.2f}"
              f"{r['size_mb']:>10.1f}{r['epochs']:>8d}"
              f"{r['train_time_min']:>11.1f}{r['test_accuracy'] * 100:>9.2f}%"
              f"{r['macro_f1']:>10.3f}{r['ms_per_image']:>9.2f}")
    print("=" * 92)


def plot_curves(histories, save_path):
    """Training and validation accuracy of all models in one figure.
    The legend shows the accuracy of the last epoch of each model."""

    panels = [
        ("accuracy", "Accuracy (%)", "(a) Training accuracy of models"),
        ("val_accuracy", "Validation Accuracy (%)",
         "(b) Validation accuracy of models"),
    ]
    markers = ["o", "^", "s", "D", "x", "+"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (key, ylabel, title) in zip(axes, panels):
        for i, (name, history) in enumerate(histories.items()):
            values = np.array(history[key]) * 100
            ax.plot(values, marker=markers[i % len(markers)], markersize=4,
                    label=f"{name}: {values[-1]:.2f}%")

        ax.set_title(title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def save_results(results, n_test):

    csv_path = os.path.join(OUTPUT_DIR, "comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved: {csv_path}")

    names = [r["model"] for r in results]
    panels = [
        ("Test accuracy (%)", [r["test_accuracy"] * 100 for r in results]),
        ("Parameters (million)", [r["params_million"] for r in results]),
        ("Training time (min)", [r["train_time_min"] for r in results]),
        ("Inference (ms / image)", [r["ms_per_image"] for r in results]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (title, values) in zip(axes, panels):
        bars = ax.bar(names, values, color="tab:blue")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.1f}" if v >= 10 else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)

    # Zoom the accuracy axis so small differences are visible
    low = min(panels[0][1])
    axes[0].set_ylim(max(0, low - 5), 100)
    axes[0].set_xlabel(f"(axis starts at {max(0, low - 5):.0f}%; "
                       f"{n_test} test images)")

    fig.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "comparison.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {png_path}")


def main():

    args = parse_args()

    print("--" * 30)
    print("Image Recognition: Rock Paper Scissors (model comparison)")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Dataset
    print("\n[Step 1] Loading dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test, classes = load_splits(
        args.data_path, args.img_size, args.max_per_class
    )

    # The class list (index = label) is saved for test_vgg.py
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    if set(classes) != EXPECTED_CLASSES:
        print(f"WARNING: expected classes {sorted(EXPECTED_CLASSES)}, "
              f"found {classes}")

    print("\nDataset loaded successfully.")
    print(f"Classes      : {classes}")

    # Step 2: Preprocessing (resize + BGR->RGB already done while loading)
    print("\n[Step 2] Preprocessing images...")
    print(f"Feature shape: {X_train.shape[1:]}")

    # Step 3: Split Dataset (all models use this same split)
    print("\n[Step 3] Splitting dataset...")
    print_split_summary(
        {"train": y_train, "val": y_val, "test": y_test}, classes
    )

    # The test set is saved for test_vgg.py
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    data = (X_train, y_train, X_val, y_val)

    # Step 4: Train and evaluate every model
    results = []
    histories = {}
    for name in args.models:
        print("\n" + "=" * 60)
        print(f"[Step 4] Model: {name}")
        print("=" * 60)

        # Same seed for every model
        keras.utils.set_random_seed(SEED)

        try:
            start = time.perf_counter()
            model, history = MODELS[name](
                data, len(classes), args.epochs, args.batch_size,
                args.head_epochs
            )
            train_time = time.perf_counter() - start
            print("Training completed.")

            # Step 5: prediction and evaluation on the test set
            print(f"\n[Step 5] Evaluating {name}...")
            results.append(evaluate_one(
                name, model, history, train_time, X_test, y_test, classes
            ))
            histories[name] = history
        except Exception as error:
            # One failing model (e.g. weights cannot be downloaded) should
            # not throw away the results of the others
            print(f"FAILED: {name}: {error}")

        keras.backend.clear_session()

    if not results:
        print("\nNo model finished.")
        return

    # Step 6: Compare the models
    print("\n[Step 6] Comparing models...")
    print_table(results)
    save_results(results, len(X_test))
    plot_curves(histories, os.path.join(OUTPUT_DIR, "accuracy_curves.png"))


if __name__ == "__main__":
    main()