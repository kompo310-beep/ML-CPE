import json
import os
from types import SimpleNamespace

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
# Pretrained weights are downloaded once by Keras (needs internet)
WEIGHTS = "imagenet"
@keras.utils.register_keras_serializable(package="RPS")
def vgg16_preprocess(images):
    # แปลง RGB เป็น BGR และลบค่าเฉลี่ย ImageNet ตามรูปแบบ VGG16
    images = tf.reverse(images, axis=[-1])
    means = tf.constant(
        [103.939, 116.779, 123.68],
        dtype=images.dtype,
    )
    return images - means

def build_transfer_model(backbone, scaling, input_shape, num_classes):
    """Pretrained backbone + the same augmentation and kind of head as the
    VGG model. Returns (model, backbone_layer).

    scaling: "tf"  -> backbone expects pixels in [-1, 1] (Rescaling added)
             "raw" -> backbone rescales by itself, expects 0-255
    """

    base = backbone(include_top=False, weights=WEIGHTS,
                    input_shape=input_shape)

    inputs = keras.Input(shape=input_shape)

    # Augmentation, active during fit() only
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)

    # Scale pixels the way this backbone was pretrained, inside the model
    if scaling == "tf":
        x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
    elif scaling == "caffe":
        x = layers.Lambda(
            vgg16_preprocess,
            name="vgg16_preprocess",
        )(x)
    elif scaling != "raw":
        raise ValueError(f"Unknown input scaling: {scaling}")
    
    # training=False keeps the BatchNorm layers of the backbone in inference
    # mode, also while fine-tuning (standard transfer-learning practice)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs), base


def compile_model(model, learning_rate):

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def train_transfer_model(name, backbone, scaling, X_train, y_train,
                         X_val, y_val, num_classes, output_dir=None,
                         epochs=30, batch_size=32, head_epochs=5):

    model, base = build_transfer_model(
        backbone, scaling, X_train.shape[1:], num_classes
    )
    model.summary()

    # Stage 1: backbone frozen, only the new head is trained
    base.trainable = False
    compile_model(model, 1e-3)
    print(f"\nStage 1: head only ({head_epochs} epochs)")
    h1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=head_epochs,
        batch_size=batch_size,
        verbose=1,
    )

    # Stage 2: unfreeze everything and fine-tune with a small learning rate
    base.trainable = True
    compile_model(model, 1e-5)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True,
            start_from_epoch=5
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
        ),
    ]

    print("\nStage 2: fine-tuning the whole network")
    h2 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # One history over both stages, with the same keys as the VGG history
    keys = ("accuracy", "val_accuracy", "loss", "val_loss")
    history = SimpleNamespace(
        history={k: h1.history[k] + h2.history[k] for k in keys}
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        model.save(os.path.join(output_dir, f"{name}.keras"))
        with open(os.path.join(output_dir, f"{name}_history.json"), "w") as f:
            json.dump({k: [float(v) for v in vs]
                       for k, vs in history.history.items()}, f)

        print(f"Saved: {os.path.join(output_dir, f'{name}.keras')}")

    return model, history


def predict_model(model, X_test):

    # One probability per class; the predicted label is the largest one
    probabilities = model.predict(X_test, verbose=0)

    return probabilities.argmax(axis=1)