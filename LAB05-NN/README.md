# Neural Network on Tomato Leaf Disease (PlantVillage)
This lab builds a simple Neural Network (MLP) to classify tomato leaf diseases from leaf images, using the PlantVillage Tomato Leaf Dataset, which contains 10 classes (9 diseases + healthy). It covers the full pipeline: loading and preprocessing image data, splitting the dataset into train/validation/test sets, training the model, and evaluating performance with accuracy, a classification report, and a confusion matrix.

# Data
PlantVillage Tomato Leaf Dataset (10 classes: 9 diseases + healthy):
https://www.kaggle.com/datasets/charuchaudhry/plantvillage-tomato-leaf-dataset?select=plantvillage

Download and extract it so the class folders sit directly under `tomato-dataset/`, e.g.:
`tomato-dataset/Tomato___Early_blight/`, `tomato-dataset/Tomato___healthy/`, etc.
(Folder names are detected automatically, so this works even if the exact
class names differ slightly from one Kaggle mirror to another.)

# Requirements

numpy
opencv-python
scikit-learn
matplotlib
tensorflow

# Structure

ML-06-NN/
│
├── tomato-dataset/
│   ├── Tomato___Bacterial_spot/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   │
│   ├── Tomato___Early_blight/
│   │   └── ...
│   │
│   ├── Tomato___Late_blight/
│   ├── Tomato___Leaf_Mold/
│   ├── Tomato___Septoria_leaf_spot/
│   ├── Tomato___Spider_mites Two-spotted_spider_mite/
│   ├── Tomato___Target_Spot/
│   ├── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
│   ├── Tomato___Tomato_mosaic_virus/
│   └── Tomato___healthy/
│
├── classification/
    ├── main.py                     # Runs the full pipeline
    ├── data_loader.py              # Loads images from class folders
    ├── preprocessing.py            # Resizes and converts BGR to RGB
    ├── split_data.py               # Splits into train/val/test sets
    ├── nn_model.py                 # Builds, trains, and predicts with the NN
    ├── evaluate.py                 # Accuracy, report, and plots
    ├── test_nn.py                  # Tests the model on sample images
    └── outputs/                    # Saved arrays, model, and plots
        ├── features.npy
        ├── labels.npy
        ├── classes.json
        ├── X_train.npy
        ├── X_val.npy
        ├── X_test.npy
        ├── y_train.npy
        ├── y_val.npy
        ├── y_test.npy
        ├── nn_model.keras
        ├── history.json
        ├── confusion_matrix.png
        ├── training_history.png
        └── prediction_sample.png