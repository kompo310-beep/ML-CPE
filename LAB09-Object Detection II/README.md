# LAB09 — Object Detection II: Toy Car Parking Detection

## Lab Summary

This lab trains a YOLO11n object detection model to detect **toy cars** in a miniature parking lot. It uses the Toy Car Parking Detection dataset, which has one class: `vehicle`. Labels use YOLO bounding boxes.

> The model detects toy cars. It does not classify parking spaces as empty or occupied. Its performance on real cars may be poor because real-world images differ from the training data.

## Dataset

The dataset has two splits:

| Split | Images | Location |
|---|---:|---|
| Train | 425 | `dataset/train/images/` |
| Validation | 75 | `dataset/valid/images/` |

Each image has a matching `.txt` label file with the same filename, stored in the corresponding `labels/` folder. Each label line follows the normalized YOLO format:

```text
class_id x_center y_center width height
```

Coordinates are normalized to the range 0–1. The `dataset/data.yaml` file defines the class as `0: vehicle` and specifies the dataset paths.

## Project Structure

```text
LAB09-Object Detection II/
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   ├── data.yaml
│   └── DATASET_README.md
├── runs/
│   ├── parking/       # Training results, plots, and model weights
│   ├── evaluation/    # Validation plots
│   └── image_test/    # Sample images with predicted boxes
├── main.py
├── train.py
├── evaluate.py
├── image_test.py
├── realtime_test.py
├── requirements.txt
└── README.md
```

## Environment and Installation

Use the existing Python environment with the packages listed in `requirements.txt`. You do not need to create a new environment.

Check the installed Ultralytics version:

```powershell
python -c "import ultralytics; print(ultralytics.__version__)"
```

If a required package is missing, install the requirements in the active environment:

```powershell
python -m pip install -r requirements.txt
```

## Run the Complete Workflow

Open PowerShell in the project folder and run:

```powershell
python main.py
```

Before running, edit `EPOCHS = 50` near the top of `main.py` to set the number of training epochs. The program does not prompt for the epoch value.

### Workflow

1. **`train.py`** — Trains YOLO11n using the pretrained weights `yolo11n.pt`, image size 640, batch size 4, and seed 42. Results are saved in `runs/parking/`.
2. **`evaluate.py`** — Loads `runs/parking/weights/best.pt` and evaluates it on the validation split. It reports Precision, Recall, mAP50, and mAP50–95, and saves plots in `runs/evaluation/`.
3. **`image_test.py`** — Selects up to five validation images and saves images with predicted bounding boxes in `runs/image_test/`.
4. **`realtime_test.py`** — Opens camera index 0 for real-time detection. Press **Q** in the camera window to close the camera and exit.

`main.py` stops if any step fails. The webcam test runs last.

## Recorded Training Results

According to `runs/parking/results.csv`, training ran for 50 epochs on CPU with PyTorch `2.14.1+cpu` and Ultralytics `8.4.158`. Total training time was approximately **5,122 seconds (85 minutes)**.

Epoch 43 had the highest mAP50–95 in the training results:

| Metric | Value at epoch 43 |
|---|---:|
| Precision | 0.8464 |
| Recall | 0.8798 |
| mAP50 | 0.8985 |
| mAP50–95 | 0.8448 |

At epoch 50, Precision was 0.8366, Recall was 0.9114, mAP50 was 0.8834, and mAP50–95 was 0.8318. These values come from validation during training; they are not results from a separate test set, so treat them as preliminary reference results.

The best model weights are saved at `runs/parking/weights/best.pt`. The final epoch weights are saved as `last.pt`.

## Key Output Files

- `runs/parking/results.csv` — Training and validation metrics for each epoch
- `runs/parking/results.png` — Training curves
- `runs/parking/confusion_matrix.png` — Confusion matrix
- `runs/parking/weights/best.pt` — Best model checkpoint based on validation results
- `runs/evaluation/` — Validation plots and images from evaluation
- `runs/image_test/` — Validation images with predicted bounding boxes

## Limitations and Discussion

- The validation split contains 75 images, so results may vary across datasets and conditions.
- The model has one class and detects toy car locations; it does not directly determine whether a parking space is empty or occupied.
- Webcam results are most relevant when using toy cars and a camera view similar to the training images.
- Performance on real cars or real parking lots may differ because the image domain is different.
- This project evaluates on the validation split and does not have a separate test split.

## Dataset Source

Toy Car Parking Detection Dataset on Kaggle:  
https://www.kaggle.com/datasets/lockedcloud/toy-car-parking-detection

See `dataset/DATASET_README.md` for more information about the annotations and dataset license.
