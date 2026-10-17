# ML-08-Object Detection I

This project explores **YOLO11s** for car object detection using Python and the Car Object Detection dataset. It covers dataset preparation, model training, validation, and inference on testing images.

## 1. Project Overview

The project uses one pretrained YOLO11s model to detect cars. The dataset annotations are converted into YOLO format before training.

**Features:**

- Car detection using YOLO11s
- Conversion of CSV bounding-box annotations to YOLO labels
- Training and validation dataset preparation
- Model training and evaluation
- Inference on testing images
- Annotated prediction images and a submission CSV

## 2. Dataset

The [Car Object Detection dataset](https://www.kaggle.com/datasets/sshikamaru/car-object-detection) is used for training and testing.

The downloaded dataset should be extracted into the `data/` folder. It includes training images, testing images, a bounding-box CSV, and a sample submission file.

| **Split** | **Images** |
| --------- | ---------- |
| Training  | Provided in the dataset |
| Validation | Generated from the labeled training images |
| Testing   | Provided in the dataset |

**Class:** `car`

The training CSV contains bounding boxes in pixel coordinates: `xmin`, `ymin`, `xmax`, and `ymax`. The `prepare_dataset.py` script converts these annotations into YOLO format and creates the training and validation splits.

## 3. Project Structure

```text
ML-08-Object-Detection-I/
├── data/
│   ├── training_images/
│   ├── testing_images/
│   ├── train_solution_bounding_boxes (1).csv
│   └── sample_submission.csv
│
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml
│
├── outputs/
│   ├── evaluation/
│   └── training_history/
│
├── predictions/
├── runs/
├── evaluate.py
├── main.py
├── predict.py
├── prepare_dataset.py
├── train.py
├── requirements.txt
└── README.md
```

| **File or Folder** | **Purpose** |
| ------------------ | ----------- |
| `main.py` | Run dataset preparation, model training, and test-image inference in sequence |
| `prepare_dataset.py` | Convert bounding-box CSV annotations into YOLO labels and create train/validation splits |
| `train.py` | Train the YOLO11s model and save model checkpoints and training results |
| `evaluate.py` | Evaluate the trained model on the validation split and save metrics and plots |
| `predict.py` | Run inference on testing images and create annotated images and a submission CSV |
| `dataset/` | Store prepared images, labels, and the YOLO dataset configuration |
| `runs/` | Store training checkpoints, logs, and training plots |
| `outputs/` | Store evaluation results and copied training-history files |
| `predictions/` | Store annotated testing images and `submission.csv` |

## 4. Object Detection

**Run the complete workflow** — prepare the dataset, train the model, and run inference:

```powershell
python main.py
```

The number of epochs for this workflow is set by the `EPOCHS` variable in `main.py`. The current value is `40`.

**Dataset preparation** — run this separately when only preparing or rebuilding the YOLO dataset:

```powershell
python prepare_dataset.py
```

**Model training** — train the model using the prepared dataset:

```powershell
python train.py
```

When `train.py` is run directly, its default training duration is 50 epochs.

**Model evaluation** — evaluate the trained model on the validation split:

```powershell
python evaluate.py
```

The evaluation metrics and plots are saved under `outputs/evaluation/validation/`. Training-history files are copied to `outputs/training_history/`.

**Image testing** — detect cars in the testing images and save the results:

```powershell
python predict.py
```

Annotated images and `submission.csv` are saved in `predictions/`. The CSV contains the image filename and normalized bounding-box coordinates for detected cars.

The training, evaluation, and prediction scripts use the model checkpoint at `runs/yolo11s/weights/best.pt` by default. If a new training run is saved in a numbered folder such as `runs/yolo11s-2`, update the model path used by `evaluate.py` or `predict.py` to select that run's checkpoint.

## 5. References

- [Car Object Detection dataset on Kaggle](https://www.kaggle.com/datasets/sshikamaru/car-object-detection)
- [Ultralytics YOLO documentation](https://docs.ultralytics.com/)