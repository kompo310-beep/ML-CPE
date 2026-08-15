import json
import os

import joblib
import numpy as np

from data_load import load_data
from preprocess import to_features
from split_data import split_dataset
from svm_model import KERNELS, build_scaler, train_svm, predict_svm
from evaluate import evaluate_model, plot_kernel_comparison

# Point this at the folder that contains one sub-folder per Tomato
# class after extracting the Kaggle PlantVillage Tomato-leaf dataset
#   tomato-dataset/
#     Tomato___Bacterial_spot/
#     Tomato___Early_blight/
#     Tomato___Late_blight/
#     Tomato___Leaf_Mold/
#     Tomato___Septoria_leaf_spot/
#     Tomato___Spider_mites Two-spotted_spider_mite/
#     Tomato___Target_Spot/
#     Tomato___Tomato_Yellow_Leaf_Curl_Virus/
#     Tomato___Tomato_mosaic_virus/
#     Tomato___healthy/
DATA_PATH = "tomato-dataset"
OUTPUT_DIR = "outputs"
IMG_SIZE = 100
TEST_SIZE = 0.2
MAX_PER_CLASS = 500   # None = use all images (slow, 10 classes x ~1000-5000 each)


def main():

    print("--" * 30)
    print("SVM Image Classification: Tomato Leaf Disease")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Dataset
    print("\n[Step 1] Loading dataset...")
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)

    np.save(f"{OUTPUT_DIR}/images.npy", images)
    np.save(f"{OUTPUT_DIR}/labels.npy", labels)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Total images : {len(images)}")
    print(f"Classes      : {classes}")

    # Step 2: Preprocessing
    print("\n[Step 2] Preprocessing images...")

    X = to_features(images)
    y = labels
    print(f"Feature shape: {X.shape}")

    # Step 3: Split Dataset
    print("\n[Step 3] Splitting dataset...")

    X_train, X_test, y_train, y_test = split_dataset(X, y, TEST_SIZE)

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # Step 4: Standardize (+ PCA) once, shared by every kernel
    print("\n[Step 4] Standardizing features (Standardize + PCA)...")

    scaler, X_train_scaled = build_scaler(X_train)
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")
    print(f"Scaled feature shape: {X_train_scaled.shape}")

    # Step 5: Train + evaluate each SVM kernel
    accuracies = {}
    for kernel in KERNELS:
        print(f"\n[Step 5] Training SVM (kernel = {kernel})...")

        model = train_svm(X_train_scaled, y_train, kernel=kernel)
        joblib.dump(model, f"{OUTPUT_DIR}/svm_model_{kernel}.pkl")

        predictions = predict_svm(model, scaler, X_test)
        np.save(f"{OUTPUT_DIR}/predictions_{kernel}.npy", predictions)

        print(f"[Step 5] Evaluating SVM (kernel = {kernel})...")
        accuracy = evaluate_model(
            y_test, predictions, classes,
            save_path=f"{OUTPUT_DIR}/confusion_matrix_{kernel}.png"
        )
        accuracies[kernel] = accuracy

    # Step 6: Compare kernels
    print("\n[Step 6] Kernel comparison")
    print("------------------------------------------")
    for kernel, accuracy in accuracies.items():
        print(f"{kernel:>8}: {accuracy * 100:.2f}%")

    best_kernel = max(accuracies, key=accuracies.get)
    print(f"\nBest kernel: {best_kernel} ({accuracies[best_kernel] * 100:.2f}%)")

    with open(f"{OUTPUT_DIR}/accuracies.json", "w") as f:
        json.dump(accuracies, f, indent=2)

    plot_kernel_comparison(accuracies, f"{OUTPUT_DIR}/kernel_comparison.png")
    print(f"Saved: {OUTPUT_DIR}/kernel_comparison.png")


if __name__ == "__main__":
    main()
