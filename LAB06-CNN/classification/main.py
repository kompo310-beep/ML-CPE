import json
import os

import numpy as np

from data_loader import load_data
from preprocessing import to_features
from split_data import split_train_val
from cnn_model import train_model, predict_model
from evaluate import evaluate_model, plot_history, threshold_report

# Paths are relative to this file, so the script runs from any directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "chest_xray")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMG_SIZE = 150
VAL_SIZE = 0.15
MAX_PER_CLASS = None   # None = use all images
EPOCHS = 40
BATCH_SIZE = 64


def print_distribution(name, y, classes):
    counts = np.bincount(y, minlength=len(classes))
    text = ", ".join(f"{c}={n}" for c, n in zip(classes, counts))
    print(f"{name:<18}: {len(y)}  ({text})")


def main():

    print("--" * 30)
    print("CNN Image Recognition: Chest X-Ray (Normal vs Pneumonia)")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Dataset
    print("\n[Step 1] Loading dataset...")
    X_train_full, y_train_full, X_test, y_test, classes = load_data(
        DATA_PATH, IMG_SIZE, MAX_PER_CLASS
    )

    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Classes: {classes}")

    # Step 2: Preprocessing (grayscale + CLAHE + resize happened during
    # load; this step only reshapes to the (N, H, W, 1) the CNN expects)
    print("\n[Step 2] Preprocessing images...")

    X_train_full = to_features(X_train_full)
    X_test = to_features(X_test)

    print(f"Train feature shape: {X_train_full.shape}")
    print(f"Test feature shape : {X_test.shape}")

    # Step 3: Split Train -> Train / Validation
    print("\n[Step 3] Splitting validation set from training data...")

    X_train, X_val, y_train, y_val = split_train_val(
        X_train_full, y_train_full, VAL_SIZE
    )

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_val.npy", X_val)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_val.npy", y_val)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print_distribution("Training samples", y_train, classes)
    print_distribution("Validation samples", y_val, classes)
    print_distribution("Testing samples", y_test, classes)

    # Step 4: Train Model
    print("\n[Step 4] Training model...")

    model, history = train_model(
        X_train, y_train, X_val, y_val,
        OUTPUT_DIR, EPOCHS, BATCH_SIZE
    )

    print("Training completed.")

    # Step 5: Prediction
    print("\n[Step 5] Testing model...")
    predictions, probabilities = predict_model(model, X_test)

    # Step 6: Evaluation
    print("\n[Step 6] Evaluating model...")
    evaluate_model(
        y_test, predictions, probabilities, classes,
        cm_path=f"{OUTPUT_DIR}/confusion_matrix.png",
        roc_path=f"{OUTPUT_DIR}/roc_curve.png",
    )
    threshold_report(y_test, probabilities)
    plot_history(history, f"{OUTPUT_DIR}/training_history.png")


if __name__ == "__main__":
    main()