from tensorflow.keras import applications

from transfer_learning import (
    build_transfer_model, train_transfer_model, predict_model
)

NAME = "mobilenetv2"
BACKBONE = applications.MobileNetV2

# MobileNetV2 expects pixels in [-1, 1]
SCALING = "tf"


def build_model(input_shape, num_classes):

    model, _ = build_transfer_model(BACKBONE, SCALING, input_shape, num_classes)

    return model


def train_model(X_train, y_train, X_val, y_val, num_classes,
                output_dir=None, epochs=30, batch_size=32, head_epochs=5):

    return train_transfer_model(
        NAME, BACKBONE, SCALING, X_train, y_train, X_val, y_val,
        num_classes, output_dir, epochs, batch_size, head_epochs
    )


# predict_model is imported above: same interface as vgg_model.py