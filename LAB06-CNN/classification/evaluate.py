import matplotlib

# Set backend before pyplot, so it works without a display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(y_test, predictions, probabilities, classes,
                   cm_path=None, roc_path=None):

    # Pin label order so target_names always matches the columns
    labels = list(range(len(classes)))

    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    print("\n------------ Evaluation ------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"ROC AUC : {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=classes,
        zero_division=0
    ))

    matrix = confusion_matrix(y_test, predictions, labels=labels)
    print("Confusion Matrix:")
    print(matrix)

    # Medical-style metrics. Positive class = last class (PNEUMONIA)
    tn, fp, fn, tp = matrix.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0   # recall of positive
    specificity = tn / (tn + fp) if (tn + fp) else 0.0   # recall of negative
    print(f"\nSensitivity (catch {classes[1]}): {sensitivity * 100:.2f}%")
    print(f"Specificity (catch {classes[0]}): {specificity * 100:.2f}%")

    if cm_path:
        plot_confusion_matrix(matrix, classes, cm_path)
        print(f"Saved: {cm_path}")

    if roc_path:
        plot_roc(y_test, probabilities, auc, roc_path)
        print(f"Saved: {roc_path}")

    return accuracy


def plot_confusion_matrix(matrix, classes, save_path):

    fig, ax = plt.subplots(figsize=(5, 5))
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


def plot_roc(y_test, probabilities, auc, save_path):

    fpr, tpr, _ = roc_curve(y_test, probabilities)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"CNN (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_history(history, save_path):
    """Accuracy, loss and learning-rate curves."""

    h = history.history

    # Keras 3 logs "learning_rate", older versions log "lr"
    lr_key = next((k for k in ("learning_rate", "lr") if k in h), None)

    n_plots = 3 if lr_key else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 4))
    epochs = range(1, len(h["loss"]) + 1)

    axes[0].plot(epochs, h["accuracy"], label="train")
    axes[0].plot(epochs, h["val_accuracy"], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, h["loss"], label="train")
    axes[1].plot(epochs, h["val_loss"], label="validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].legend()

    if lr_key:
        axes[2].plot(epochs, h[lr_key], marker="o")
        axes[2].set_yscale("log")   # LR drops by halves, log scale shows it clearly
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning rate")
        axes[2].set_title("Learning Rate")
        axes[2].grid(True, which="both", alpha=0.3)
    else:
        print("Learning rate not found in history, skipping LR plot.")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def threshold_report(y_test, probabilities, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9)):
    """Show how sensitivity/specificity trade off as the threshold moves.

    Analysis only: don't pick a threshold from this table and report the
    resulting test-set score as the model's headline result — that would
    mean tuning on the test set.
    """

    y_test = np.asarray(y_test)

    print("\nThreshold sweep (analysis only)")
    print(f"{'thr':>5} {'acc':>8} {'sens':>8} {'spec':>8}")

    for t in thresholds:
        pred = (probabilities > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        acc = (tp + tn) / len(y_test)
        print(f"{t:>5.1f} {acc * 100:>7.2f}% {sens * 100:>7.2f}% {spec * 100:>7.2f}%")