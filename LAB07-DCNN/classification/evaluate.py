import matplotlib

# Set backend before pyplot, so it works without a display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def confusion_matrix(y_true, y_pred, n_classes):
    # rows = true class, columns = predicted class
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def classification_report(matrix, classes):
    tp = np.diag(matrix).astype(float)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)

    precision = np.divide(tp, predicted, out=np.zeros_like(tp),
                          where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp),
                       where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp),
                   where=denom > 0)

    width = max(12, max(len(c) for c in classes) + 2)
    lines = [f"{'':>{width}}{'precision':>10}{'recall':>10}"
             f"{'f1-score':>10}{'support':>10}", ""]

    for i, name in enumerate(classes):
        lines.append(f"{name:>{width}}{precision[i]:>10.2f}{recall[i]:>10.2f}"
                     f"{f1[i]:>10.2f}{support[i]:>10d}")

    total = support.sum()
    accuracy = tp.sum() / total
    weights = support / total

    lines.append("")
    lines.append(f"{'accuracy':>{width}}{'':>10}{'':>10}"
                 f"{accuracy:>10.2f}{total:>10d}")
    lines.append(f"{'macro avg':>{width}}{precision.mean():>10.2f}"
                 f"{recall.mean():>10.2f}{f1.mean():>10.2f}{total:>10d}")
    lines.append(f"{'weighted avg':>{width}}"
                 f"{(precision * weights).sum():>10.2f}"
                 f"{(recall * weights).sum():>10.2f}"
                 f"{(f1 * weights).sum():>10.2f}{total:>10d}")

    return "\n".join(lines)


def evaluate_model(y_test, predictions, classes, save_path=None):

    y_test = np.asarray(y_test)
    predictions = np.asarray(predictions)

    # Class order is fixed by the label numbers, same as classes.json
    matrix = confusion_matrix(y_test, predictions, len(classes))

    accuracy = float(np.mean(y_test == predictions))

    print("\n------------ Evaluation ------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(matrix, classes))
    print()

    print("Confusion Matrix:")
    print(matrix)

    if save_path:
        plot_confusion_matrix(matrix, classes, save_path)
        print(f"Saved: {save_path}")

    return accuracy


def plot_confusion_matrix(matrix, classes, save_path):

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(np.arange(len(classes)), classes)
    ax.set_yticks(np.arange(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    threshold = matrix.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_history(history, save_path):

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")