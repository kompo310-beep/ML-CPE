from tensorflow.keras import applications

from transfer_learning import (
    build_transfer_model,
    train_transfer_model,
    predict_model,
)

NAME = "vgg16"
BACKBONE = applications.VGG16

# VGG16 ใช้การ preprocess แบบ Caffe
SCALING = "caffe"


def build_model(input_shape, num_classes):
    model, _ = build_transfer_model(
        BACKBONE, SCALING, input_shape, num_classes
    )
    return model


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes,
    output_dir=None,
    epochs=30,
    batch_size=32,
    head_epochs=5,
):
    return train_transfer_model(
        NAME,
        BACKBONE,
        SCALING,
        X_train,
        y_train,
        X_val,
        y_val,
        num_classes,
        output_dir,
        epochs,
        batch_size,
        head_epochs,
    )