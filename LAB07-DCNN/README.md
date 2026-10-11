# ML-07-Deep Convolutional Neural Network (DCNN)

Build a DCNN based on the VGG architecture using Python for image classification. The project covers image loading, preprocessing, dataset splitting, DCNN (VGG) model training, evaluation, and Rock Paper Scissors hand-gesture classification.

# Data

Kaggle Rock Paper Scissors: https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors

The `rps-cv-images` folder holds 2,188 photos (300x200, green background) of hand gestures in three classes: `paper`, `rock`, `scissors`.

## Download and setup

1. Download the dataset from the link above (button **Download**, gives `archive.zip`), or with the Kaggle CLI:

   ```bash
   kaggle datasets download -d drgfreeman/rockpaperscissors -p rockpaperscissors --unzip
   ```

2. Extract it into a folder named `rockpaperscissors/` next to `classification/` (see the structure below).
3. Install the requirements:

   ```bash
   pip install -r requirements.txt
   ```


# Structure

```text
ML-08-DCNN/
│
├── rockpaperscissors/                 # extracted Kaggle dataset
│   ├── README_rpc-cv-images.txt       # dataset notes (not used by the code)
│   ├── rps-cv-images/                 # <- used by default
│   │   ├── paper/
│   │   │   ├── xxxx.png
│   │   │   └── ...
│   │   ├── rock/
│   │   └── scissors/
│   ├── rps/                           # other copies in the archive, ignored
│   └── rps-test-set/
│
├── classification/
│   ├── main.py                     # Main training pipeline
│   ├── data_loader.py              # Find the dataset, load images, skip corrupted/duplicate files
│   ├── preprocessing.py            # Resize images and convert BGR to RGB
│   ├── split_data.py               # Stratified training, validation, and test split
│   ├── vgg_model.py                # Build, train, save, and predict using the VGG model
│   ├── evaluate.py                 # Accuracy, classification report, confusion matrix, and training history plots
│   ├── test_vgg.py                 # Predict Rock / Paper / Scissors for sample images
│   │
│   └── outputs/                    # Generated files
│       ├── classes.json            # class names, index = label
│       ├── X_test.npy
│       ├── y_test.npy
│       ├── vgg_model.keras
│       ├── history.json
│       ├── confusion_matrix.png
│       ├── training_history.png
│       └── prediction_sample.png
│
└── requirements.txt
```

# How to run

Run from the `classification/` folder (any folder works, paths are relative to the scripts):

```bash
cd classification

# Train with the default settings at the top of main.py
python main.py

# Choose the VGG layout: vgg16 (13 conv layers) or small (10 conv layers, faster on CPU)
python main.py --model vgg16
python main.py --model small

```

Prediction after training:

```bash
# 4 random images from the saved test set
python test_vgg.py

# more random test images
python test_vgg.py -n 9

# your own images (true class is read from the parent folder name if it is
# paper / rock / scissors, otherwise shown as unknown)
python test_vgg.py ../rockpaperscissors/rps-cv-images/rock/abc.png my_hand.jpg
```

The predicted class and confidence are printed (with the score of every class) and saved as `outputs/prediction_sample.png`.

Settings you can change at the top of `main.py`: `IMG_SIZE` (100), `TEST_SIZE` (0.2), `VAL_SIZE` (0.1), `MAX_PER_CLASS` (None = all), `EPOCHS` (30), `BATCH_SIZE` (32), `MODEL` (`vgg16` or `small`).

# DCNN workflow

1. **Load** - `data_loader.py` finds the class folders, reads every image, converts BGR to RGB, resizes to `IMG_SIZE x IMG_SIZE`, and skips unreadable files and exact duplicate files (with a count per class).
2. **Split** - `split_data.py` makes a stratified train / validation / test split (70 / 10 / 20 by default), so each split keeps the class proportions. Images are split by index and checked for overlap, so none is in two splits. Ready-made `train/val/test` folders are used as they are.
3. **Model** - `vgg_model.py` stacks VGG blocks (Conv3x3 + BatchNorm + ReLU, then max-pooling), then global average pooling, two dense layers with dropout, and a 3-unit softmax output. Pixel scaling and light augmentation (flip, rotation, zoom; training only) are layers inside the model. `vgg16` is the 13-conv layout, `small` is a lighter one.
4. **Train** - Adam with `sparse_categorical_crossentropy` (labels are integers), early stopping (checked from epoch 6 onward) and learning-rate reduction on the validation loss. The best weights are kept.
5. **Evaluate** - the test set is used once: accuracy, per-class precision/recall/F1, and a confusion matrix.
6. **Predict** - `test_vgg.py` loads the saved model and `classes.json` and shows the predicted class with its confidence.

# Summary

This project implements a VGG-based DCNN for Rock, Paper, and Scissors image classification. It includes image loading, preprocessing, stratified dataset splitting, model training, evaluation, and prediction using a modular Python pipeline.
