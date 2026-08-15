import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "outputs"
IMG_SIZE = 100
N_SAMPLES = 4
KERNEL = "rbf"  # which trained kernel to demo: "linear", "poly", or "rbf"


def test_svm(kernel=KERNEL, n_samples=N_SAMPLES):

    # Load model, shared scaler, and test set
    model = joblib.load(f"{OUTPUT_DIR}/svm_model_{kernel}.pkl")
    scaler = joblib.load(f"{OUTPUT_DIR}/scaler.pkl")
    X_test = np.load(f"{OUTPUT_DIR}/X_test.npy")
    y_test = np.load(f"{OUTPUT_DIR}/y_test.npy")
    with open(f"{OUTPUT_DIR}/classes.json") as f:
        classes = json.load(f)

    # Pick random images (no seed -> different every run)
    index = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[index]
    y_sample = y_test[index]