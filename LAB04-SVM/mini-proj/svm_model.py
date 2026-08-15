from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# The three kernels the assignment asks us to compare
KERNELS = ("linear", "poly", "rbf")

# Per-kernel hyperparameters. Poly/RBF get a small PCA-whitened feature
# space so training stays fast; linear SVMs handle the higher-dim raw
# pixels reasonably well but we keep the same input for a fair,
# apples-to-apples comparison across kernels.
_KERNEL_PARAMS = {
    "linear": dict(kernel="linear", C=1),
    "poly": dict(kernel="poly", degree=3, C=1, gamma="scale", coef0=1),
    "rbf": dict(kernel="rbf", C=10, gamma="scale"),
}


def build_scaler(X_train, pca_components=150):
    """Fit Standardize + PCA once on the training data.

    Every kernel is trained/evaluated on this same transformed
    feature space, so accuracy differences come from the kernel
    choice, not from different preprocessing.
    """
    scaler = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(pca_components, *X_train.shape),
                    whiten=True, random_state=42)),
    ])

    X_train_scaled = scaler.fit_transform(X_train)

    return scaler, X_train_scaled


def train_svm(X_train_scaled, y_train, kernel="rbf"):
    """Train one SVM with the given kernel on already-scaled features."""

    if kernel not in _KERNEL_PARAMS:
        raise ValueError(f"Unknown kernel '{kernel}', expected one of {KERNELS}")

    model = SVC(cache_size=1000, **_KERNEL_PARAMS[kernel])
    model.fit(X_train_scaled, y_train)

    return model


def predict_svm(model, scaler, X_test):
    # Apply the same Standardize + PCA transform used for training data
    X_test_scaled = scaler.transform(X_test)
    # Predict
    predictions = model.predict(X_test_scaled)

    return predictions
