# STEP 1  Load the dataset.
# STEP 2  Find the best k using the Elbow Method.
# STEP 3  Run K-Means with the selected k.
# STEP 4  Analyze each cluster.
# STEP 5  Use KNN to assign new points into clusters.

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

import data_loader
import visualize
from kmeans_tf import TFKMeans
from knn_tools import KNNClusterAssigner

OUT_DIR = Path(__file__).resolve().parent / "outputs"

N_CLUSTERS = 3      # Iris has 3 known species, a natural choice to confirm against
KNN_K = 5


def title(text):
    print("\n" + "--" * 30)
    print(text)


# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True)

    title("STEP 1 : load data")
    data = data_loader.load_data()
    X = data["X"]                # scaled data (used for computation)
    X_raw = data["X_raw"]        # original units (used to explain results)
    df = data["df"]
    features = data["features"]

    print(f"size data : {X.shape[0]} rows x {X.shape[1]} features")
    print("features used for clustering :")
    for f in features:
        print(f"   - {f}")

    # =====================================================================
    title("STEP 2 : how many clusters should we use?")
    # =====================================================================
    k_values = [2, 3, 4, 5, 6, 7, 8]
    inertias = []

    for k in k_values:
        km = TFKMeans(n_clusters=k).fit(X)
        sil = silhouette_score(X, km.labels_)
        inertias.append(km.inertia_)
        print(f"   k = {k}  ->  inertia = {km.inertia_:8.1f}   silhouette = {sil:.3f}")

    visualize.plot_elbow(k_values, inertias, OUT_DIR / "01_elbow.png")
    print(f"\n plot saved to outputs/01_elbow.png, use the bend in the curve to pick k")
    print(f" selected k = {N_CLUSTERS}")

    # =====================================================================
    title(f"STEP 3 : Run K-Means (k = {N_CLUSTERS})")
    # =====================================================================
    km = TFKMeans(n_clusters=N_CLUSTERS)
    labels = km.fit_predict(X)

    sil = silhouette_score(X, labels)
    print(f"used {km.n_iter_} iterations until centroids stabilized")
    print(f"Inertia          : {km.inertia_:.1f}")
    print(f"Silhouette score : {sil:.3f}")
    print(f"member count in each cluster : {np.bincount(labels).tolist()}")

    if sil < 0.25:
        print("\n[Note] A low silhouette score means weak clusters.")
        print("       This dataset may not have clear natural groups.")
        print("       K-Means always creates clusters, so always check the silhouette score.")

    # plot using the last two detected feature columns (typically petal
    # length/width in the standard Iris layout, but works for any column order)
    plot_x, plot_y = features[-2], features[-1]
    visualize.plot_clusters(X_raw[:, [-2, -1]], labels, OUT_DIR / "02_clusters.png",
                             x_name=plot_x, y_name=plot_y)

    title("STEP 4 : What are the characteristics of each cluster?")

    profile = pd.DataFrame(X_raw.astype("float64"), columns=features)
    profile["cluster"] = labels

    summary = profile.groupby("cluster").mean().round(2)
    summary["member count"] = np.bincount(labels)

    print(summary.to_string())
    summary.to_csv(OUT_DIR / "cluster_summary.csv", encoding="utf-8-sig")

    title(f"STEP 5 : use KNN to assign new flowers into clusters (k = {KNN_K})")

    # simulate a scenario: pretend we already know the cluster of most flowers,
    # and the remaining ones are "new flowers" that just arrived
    n_known = int(0.8 * len(X))
    X_known, labels_known = X[:n_known], labels[:n_known]
    X_new, labels_new = X[n_known:], labels[n_known:]

    assigner = KNNClusterAssigner(k=KNN_K)
    assigner.fit(X_known, labels_known)
    knn_pred = assigner.predict(X_new)

    accuracy = float(np.mean(knn_pred == labels_new))
    print(f"number of 'new flowers' : {len(X_new)}")
    print(f"KNN cluster assignment accuracy vs K-Means : {accuracy * 100:.1f} %")
    print("Well-separated clusters give better KNN assignment.")
    print("Use KNN for new data without rerunning K-Means.")

    # =====================================================================
    title("save results to CSV file")
    # =====================================================================
    result = df.copy()
    result["cluster"] = labels
    result.to_csv(OUT_DIR / "clustered_iris.csv",
                  index=False, encoding="utf-8-sig")

    for f in sorted(OUT_DIR.iterdir()):
        print(f"   - outputs/{f.name}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
