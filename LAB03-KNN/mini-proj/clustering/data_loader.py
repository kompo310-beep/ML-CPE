## Read data from CSV file and prepare for the clustering mini-project

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

CSV_PATH = Path(__file__).resolve().parent.parent / "dataset" / "IRIS.csv"

ID_COLUMNS = ["Id", "id", "ID"]


# ---------------------------------------------------------------------------
def load_data():
    """
    Returns a dict with
        X        : scaled data (used for clustering)
        X_raw    : data in original units (used when explaining results)
        df       : full table from the CSV file
        features : list of feature column names used
    """
    df = pd.read_csv(CSV_PATH)
    df = df.dropna()
    df = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns])

    # auto-detect features: drop the last column (species / class name),
    # use every remaining numeric column for clustering
    target_col = df.columns[-1]
    numeric_df = df.drop(columns=[target_col]).select_dtypes(include="number")
    features = list(numeric_df.columns)

    X_raw = numeric_df.to_numpy(dtype="float32")
    X = StandardScaler().fit_transform(X_raw).astype("float32")  # mean=0, std=1

    return {"X": X, "X_raw": X_raw, "df": df, "features": features}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = load_data()
    print("features used :", data["features"])
    print("size data :", data["X"].shape)
    print("mean after scale (should be close to 0) :", data["X"].mean(axis=0).round(3))
