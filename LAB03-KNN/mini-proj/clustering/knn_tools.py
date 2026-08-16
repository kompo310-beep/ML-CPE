## KNN helper used to assign new points to existing clusters (NumPy version)

import numpy as np


class KNNClusterAssigner:

    def __init__(self, k=5):
        self.k = k

    # -----------------------------------------------------------------
    def fit(self, X, cluster_labels):

        self.X = np.asarray(X, dtype=np.float32)
        self.labels = np.asarray(cluster_labels, dtype=np.int32)
        self.n_clusters = int(cluster_labels.max()) + 1
        return self

    # -----------------------------------------------------------------
    def predict(self, X_new):

        X_new = np.asarray(X_new, dtype=np.float32)

        # 1) distance from each new point to every already-labeled point
        diff = X_new[:, None, :] - self.X[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))

        # 2) pick the k closest neighbors
        idx = np.argsort(dist, axis=1)[:, :self.k]
        neighbor_labels = self.labels[idx]

        # 3) vote -> the cluster with the most votes wins
        votes = np.zeros((X_new.shape[0], self.n_clusters), dtype=np.int32)
        for c in range(self.n_clusters):
            votes[:, c] = np.sum(neighbor_labels == c, axis=1)

        return np.argmax(votes, axis=1).astype("int32")
