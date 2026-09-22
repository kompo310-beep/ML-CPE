import numpy as np
from sklearn.model_selection import train_test_split


def split_train_val(X, y, val_size=0.15):
    """Carve a validation set out of the training split.

    The test set is already provided by the dataset, so it is not touched here.
    stratify keeps the NORMAL/PNEUMONIA ratio the same in both parts.
    """

    y = np.asarray(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_size,
        random_state=42,
        stratify=y
    )

    return X_train, X_val, y_train, y_val