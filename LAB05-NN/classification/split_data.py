import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(X, y, test_size=0.2, val_size=0.1):
    """Split into train / validation / test.

    A NN needs a validation set to monitor overfitting during training.
    stratify=y keeps the class proportions balanced across the splits,
    which matters here because the 10 tomato classes are not evenly sized.
    """
    y = np.asarray(y)

    # First carve off the test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=42,
        stratify=y
    )

    # Then carve the validation set out of what remains
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio,
        random_state=42,
        stratify=y_train
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
