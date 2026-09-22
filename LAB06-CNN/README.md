# LAB06-CNN: Chest X-Ray Pneumonia Classification

A CNN pipeline that classifies chest X-ray images as NORMAL or PNEUMONIA, covering data loading, preprocessing, dataset splitting, model training, evaluation, and testing on sample images.

## Dataset

Kaggle Chest X-Ray Images (Pneumonia): https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images

Download and extract it into `chest_xray/` at the project root (`data_loader.py` automatically searches for a folder containing both `train/` and `test/`, so it still works even if the extraction nests it as `chest_xray/chest_xray/`).

## Project Structure

```text
LAB06-CNN/
│
├── chest_xray/                     # dataset from Kaggle
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
│
├── classification/
│   ├── main.py                     # main pipeline
│   ├── data_loader.py              # loads train/test images, skips corrupt files
│   ├── preprocessing.py            # grayscale + CLAHE + resize
│   ├── split_data.py               # splits train -> train/validation
│   ├── cnn_model.py                # builds, trains, saves, and predicts with the CNN
│   ├── evaluate.py                 # metrics, confusion matrix, ROC, threshold sweep, training plots
│   ├── test_cnn.py                 # tests the trained model on 4 random images
│   └── outputs/
│       ├── classes.json
│       ├── X_train.npy / X_val.npy / X_test.npy
│       ├── y_train.npy / y_val.npy / y_test.npy
│       ├── cnn_model.keras
│       ├── history.json
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── training_history.png    # accuracy, loss, learning rate
│       └── prediction_sample.png
│
└── requirements.txt
```

## How to Run

```bash
pip install -r requirements.txt
cd classification
python main.py        # load data, train, and evaluate
python test_cnn.py    # test on random images from the test set
```

## Pipeline

1. **Load** (`data_loader.py`) — reads images from `train/` and `test/` separately, using the class order detected from `train/` as the standard (`NORMAL=0`, `PNEUMONIA=1`) so labels always match across both splits
2. **Preprocess** (`preprocessing.py`) — converts to grayscale, evens out contrast with CLAHE (reduces differences between images from different X-ray machines), resizes to 150×150, reshapes into `(N, H, W, 1)` uint8
3. **Split** (`split_data.py`) — splits only the training data into train/validation with stratification (the test set provided by the dataset is left untouched)
4. **Train** (`cnn_model.py`) — a 4-block CNN (Conv+BatchNorm+MaxPool) followed by GlobalAveragePooling and a dense head; augmentation excludes horizontal flip (it would mirror the anatomy, e.g. heart position); uses class weighting to offset PNEUMONIA outnumbering NORMAL by ~3x; EarlyStopping + ReduceLROnPlateau
5. **Predict & Evaluate** (`evaluate.py`) — accuracy, precision/recall/F1, confusion matrix, ROC-AUC, sensitivity/specificity, and a threshold sweep (for analysis only — showing results at thresholds other than 0.5, not for picking a threshold from the test set)
6. **Test** (`test_cnn.py`) — samples random test-set images, predicts, and plots the results as an image grid

## Key Settings (`main.py`)

| Variable | Value | Reason |
|---|---|---|
| `IMG_SIZE` | 150 | Lung detail needs more resolution than typical images |
| `VAL_SIZE` | 0.15 | Fraction of the training data held out as validation |
| `EPOCHS` | 40 | EarlyStopping cuts training short if it plateaus |
| `BATCH_SIZE` | 64 | Helps BatchNormalization statistics stay stable |

The model uses `learning_rate=1e-4` with `clipnorm=1.0` to reduce swings in accuracy/loss during training, and `BatchNormalization(momentum=0.9)` so the moving average tracks the weights faster.

## Things to Know

- **Class order**: alphabetical, so `NORMAL=0` and `PNEUMONIA=1` — the sigmoid output is the probability of PNEUMONIA
- **Validation may look "too good"**: this dataset has multiple images per patient, so randomly splitting validation out of train can put images from the same patient on both sides, inflating validation scores. Treat the test-set results as the reliable ones
- **Gap between validation and test**: the dataset's test-set images differ somewhat from train (brightness, contrast, source), which drops specificity (catching NORMAL) on the test set well below what validation suggests. Adding CLAHE and brightness/contrast augmentation narrows this gap but doesn't close it entirely
- **threshold_report**: for analyzing the sensitivity/specificity trade-off only. Don't pick a threshold based on test-set results and report that as the headline score — that amounts to tuning on the test set
- **Class balancing**: to compare results with and without class weighting, try removing `class_weight=class_weight` from `model.fit` and see how sensitivity/specificity change

## Requirements

```text
tensorflow
numpy
opencv-python
scikit-learn
matplotlib
```