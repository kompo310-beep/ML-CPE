import numpy as np


def _stratified_split(indices, y, test_size, rng):
    """Split `indices` in two, keeping the class proportions of y."""

    first = []
    second = []

    for label in np.unique(y[indices]):
        class_idx = rng.permutation(indices[y[indices] == label])

        n_second = int(round(len(class_idx) * test_size))
        if test_size > 0 and n_second == 0 and len(class_idx) > 1:
            n_second = 1

        second.append(class_idx[:n_second])
        first.append(class_idx[n_second:])

    return (rng.permutation(np.concatenate(first)),
            rng.permutation(np.concatenate(second)))


def split_dataset(X, y, test_size=0.2, val_size=0.1, seed=42):
    y = np.asarray(y)
    rng = np.random.default_rng(seed)

    # Split indices, not images, so the split can be checked for overlap
    indices = np.arange(len(y))

    # First carve off the test set
    train_idx, test_idx = _stratified_split(indices, y, test_size, rng)

    # Then carve the validation set out of what remains
    val_ratio = val_size / (1 - test_size)
    train_idx, val_idx = _stratified_split(train_idx, y, val_ratio, rng)

    # No image may be in more than one split
    if (set(train_idx) & set(val_idx)
            or set(train_idx) & set(test_idx)
            or set(val_idx) & set(test_idx)):
        raise RuntimeError("Data leakage: an image is in more than one split")

    return (X[train_idx], X[val_idx], X[test_idx],
            y[train_idx], y[val_idx], y[test_idx])


def print_split_summary(splits, classes):
    """splits = {"train": y_train, "val": y_val, "test": y_test}"""

    for name, y in splits.items():
        counts = np.bincount(y, minlength=len(classes))
        per_class = ", ".join(
            f"{c}: {n} ({n / len(y) * 100:.1f}%)"
            for c, n in zip(classes, counts)
        )
        print(f"{name:<6}: {len(y):>5} images | {per_class}")