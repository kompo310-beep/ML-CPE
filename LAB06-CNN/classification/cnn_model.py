import json
import os

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def conv_block(filters):
    """Conv -> BatchNorm -> MaxPool. momentum=0.9 lets the BN moving
    statistics track the weights faster, which stabilises validation."""

    return [
        layers.Conv2D(filters, 3, padding="same", activation="relu"),
        layers.BatchNormalization(momentum=0.9),
        layers.MaxPooling2D(),
    ]


def build_model(input_shape):
    """CNN for binary X-ray classification: 4 conv blocks -> dense head."""

    model = keras.Sequential([
        keras.Input(shape=input_shape),

        # Normalize 0-255 to 0-1 inside the model
        layers.Rescaling(1.0 / 255),

        # Light augmentation, only active during training.
        # No horizontal flip: it would mirror the anatomy (heart side).
        # Brightness/contrast jitter imitates different X-ray machines.
        layers.RandomBrightness(0.15, value_range=(0, 1)),
        layers.RandomContrast(0.15),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.05, 0.05),

        *conv_block(32),
        *conv_block(64),
        *conv_block(128),
        *conv_block(256),

        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),

        # 1 sigmoid output = probability of the positive class (PNEUMONIA)
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        # Lower LR + gradient clipping = smoother training curves
        optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def compute_class_weights(y):
    """Give the rarer class a larger weight so the model is not biased
    toward the majority class (PNEUMONIA is ~3x more common)."""

    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    return {
        int(c): total / (len(classes) * n)
        for c, n in zip(classes, counts)
    }


def train_model(X_train, y_train, X_val, y_val,
                output_dir=None, epochs=40, batch_size=64):
    """Build, train and save the model. Returns (model, history).

    Defaults match main.py's EPOCHS/BATCH_SIZE; both are still
    overridable by whoever calls this function directly.
    """

    model = build_model(X_train.shape[1:])
    model.summary()

    class_weight = compute_class_weights(y_train)
    print("\nClass weights:", class_weight)

    callbacks = [
        # Stop when validation loss stops improving, keep the best weights
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        # Halve the LR on plateau; also logs it into history for the LR plot
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    print("\nTraining...")
    history = model.fit(
        X_train, y_train.astype("float32"),
        validation_data=(X_val, y_val.astype("float32")),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, "cnn_model.keras")
        model.save(model_path)

        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump({k: [float(v) for v in vs]
                       for k, vs in history.history.items()}, f)

        print(f"Saved: {model_path}")

    return model, history


def predict_model(model, X_test, threshold=0.5):
    """Returns (predicted labels, probability of the positive class)."""

    probabilities = model.predict(X_test, verbose=0).ravel()
    predictions = (probabilities > threshold).astype(int)

    return predictions, probabilities