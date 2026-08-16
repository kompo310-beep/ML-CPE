# ML-04-K-Nearest Neighbors (KNN) on Iris

Build a KNN + K-Means pipeline using Python and NumPy, including data loading,
preprocessing, feature scaling, model training, evaluation, and prediction.

# Data
Iris Dataset (IRIS.csv) — sepal/petal length & width, with species as the last
column (target). Column names are auto-detected, so this works with any of the
common Iris CSV variants (e.g. `SepalLengthCm`/`Species` or
`sepal_length`/`species`).
Dataset source: [Kaggle - Iris Flower Dataset](https://www.kaggle.com/code/drisrarahmad/iris-flower-dataset)

# Requirements
```text
numpy
pandas
scikit-learn
matplotlib
```
(No TensorFlow needed — the KNN/K-Means models are implemented with NumPy.)

# Structure
```text
mini-proj/
│
├── dataset/
│   └── IRIS.csv
│
├── classification/
│   ├── main.py
│   ├── data_loader.py
│   ├── knn_tf.py
│   ├── evaluate.py
│   └── outputs/
│       ├── 01_k_curve.png
│       ├── 02_confusion_matrix.png
│       └── predictions.csv
│
├── clustering/
│   ├── main.py
│   ├── data_loader.py
│   ├── kmeans_tf.py
│   ├── knn_tools.py
│   ├── visualize.py
│   └── outputs/
│       ├── 01_elbow.png
│       ├── 02_clusters.png
│       ├── cluster_summary.csv
│       └── clustered_iris.csv
│
└── requirements.txt
```

# Summary
This project demonstrates KNN for classification and K-Means (with a KNN cluster
assigner) for clustering, both implemented from scratch with NumPy and
cross-checked against scikit-learn, using the Iris dataset. It includes data
loading, preprocessing, model training, evaluation, visualization, and prediction
through a modular Python pipeline.
