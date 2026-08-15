# SVM on Tomato Leaf Disease (PlantVillage)

This project uses Support Vector Machine (SVM) to classify tomato leaf diseases from the
PlantVillage Tomato Leaf dataset,
which consists of 10 classes (9 diseases + healthy leaves). The file structure remains the same, but the code has been rewritten to:

Handle a multi-class dataset

Train and compare three SVM kernels: Linear, Polynomial, and RBF

## Files

| File            | Purpose                                                             |
|-----------------|----------------------------------------------------------------------|
| `data_load.py`  | Loads images from `tomato-dataset/<class>/*.jpg`, auto-detects classes from folder names |
| `preprocess.py` | Grayscale + resize each image, flatten to a normalized feature vector |
| `split_data.py` | Stratified train/test split                                        |
| `svm_model.py`  | Standardize + PCA (shared across kernels), train one SVM per kernel |
| `evaluate.py`   | Accuracy, classification report, confusion matrix, kernel comparison chart |
| `main.py`       | Runs the full pipeline: load → preprocess → split → train 3 kernels → evaluate → compare |
| `test_svm.py`   | Loads a trained kernel's model and visualizes predictions on random test samples |

## Dataset setup

1. Download the dataset from Kaggle (link above) — you'll need a free
   Kaggle account and the `kaggle` CLI, or the "Download" button on
   the dataset page.
2. Extract it so you end up with one folder per class under
   `tomato-dataset/`, e.g.:

   ```
   tomato-dataset/
     Tomato___Bacterial_spot/
     Tomato___Early_blight/
     Tomato___Late_blight/
     Tomato___Leaf_Mold/
     Tomato___Septoria_leaf_spot/
     Tomato___Spider_mites Two-spotted_spider_mite/
     Tomato___Target_Spot/
     Tomato___Tomato_Yellow_Leaf_Curl_Virus/
     Tomato___Tomato_mosaic_virus/
     Tomato___healthy/
   ```

   
## Output (in `outputs/`)

- `accuracies.json` — accuracy of each kernel
- `kernel_comparison.png` — bar chart comparing the 3 kernels
- `confusion_matrix_<kernel>.png` — per-kernel confusion matrix
- `svm_model_<kernel>.pkl`, `scaler.pkl` — trained models + shared Standardize/PCA transform
- `predictions_<kernel>.npy` — predictions on the held-out test set
- `prediction_sample_<kernel>.png` — sample predictions from `test_svm.py`
