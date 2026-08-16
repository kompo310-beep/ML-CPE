"""
Read CSV
convert text to number (if any)
make Scaling for KNN
split data: train / validation / test
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CSV_PATH = Path(__file__).resolve().parent.parent / "dataset" / "IRIS.csv"

# columns to drop before building features (id-like columns, not predictive)
ID_COLUMNS = ["Id", "id", "ID"]

# text features that need to be converted to numbers (Iris normally has none
# besides the target; add here if your CSV has extra categorical columns)
TEXT_FEATURES = {}


# ---------------------------------------------------------------------------
def load_data(test_size=0.2, val_size=0.25, seed=42):

    # step 1 : read CSV
    df = pd.read_csv(CSV_PATH)
    df = df.dropna()
    df = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns])

    # step 2 : auto-detect target and feature columns
    # target = the last column in the CSV (species / class name), rest = features
    # this avoids hardcoding column names, which differ between Iris CSV versions
    target_col = df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col and c not in TEXT_FEATURES]

    X = df[feature_cols].copy()
    for col, mapping in TEXT_FEATURES.items():
        if col in df.columns:
            X[col] = df[col].map(mapping)

    # convert target to number : e.g. Iris-setosa->0, Iris-versicolor->1, ...
    class_names = sorted(df[target_col].unique())
    y = df[target_col].map({name: i for i, name in enumerate(class_names)})

    X = X.to_numpy(dtype="float32")
    y = y.to_numpy(dtype="int32")

    # step 3 : split data into train 60 / validation 20 / test 20
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=seed, stratify=y_temp)

    # step 4 : Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_val = scaler.transform(X_val).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "class_names": class_names,
        "feature_names": feature_cols,
        "target_name": target_col,
        "n_rows": len(df),
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = load_data()
    print("columns used as features :", data["feature_names"])
    print("column used as target    :", data["target_name"])
    print("train :", data["X_train"].shape)
    print("val   :", data["X_val"].shape)
    print("test  :", data["X_test"].shape)
    print("classes :", data["class_names"])
