## Simple K-Means Clustering implemented with NumPy (for beginners)
## Same class name/API as the TensorFlow version, so main.py needs no changes.
## centroid update step avoids empty clusters (if no member, keep centroid in place)

import numpy as np


class TFKMeans:   # K-Means (NumPy version)

    # km = TFKMeans(n_clusters=3)
    # labels = km.fit_predict(X)     # labels tell which cluster each row belongs to

    def __init__(self, n_clusters=3, max_iter=100, seed=42):
        self.n_clusters = n_clusters      # number of clusters wanted
        self.max_iter = max_iter          # max iterations
        self.seed = seed                  # random seed (fixed for reproducibility)

    # -----------------------------------------------------------------
    def _distance(self, X, centroids):    # X shape (n, d), centroids shape (k, d) -> result shape (n, k)

        diff = X[:, None, :] - centroids[None, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))

    # -----------------------------------------------------------------
    def fit(self, X):                     # Run K-Means until centroids are stable

        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]

        # step 0 : randomly pick k data points as starting centroids
        rng = np.random.default_rng(self.seed)
        start_idx = rng.choice(n_samples, size=self.n_clusters, replace=False)
        centroids = X[start_idx].copy()

        for step in range(self.max_iter):
            # step 1 : ASSIGN
            # argmin = find the closest centroid
            dist = self._distance(X, centroids)
            labels = np.argmin(dist, axis=1).astype(np.int32)

            # step 2 : UPDATE
            # average of the members in each cluster, then move the centroid
            new_centroids = np.zeros_like(centroids)
            for c in range(self.n_clusters):
                members = X[labels == c]     # points in cluster c
                if len(members) > 0:
                    new_centroids[c] = members.mean(axis=0)
                else:
                    new_centroids[c] = centroids[c]

            # check stability
            moved = float(np.max(np.abs(new_centroids - centroids)))
            centroids = new_centroids
            if moved < 1e-4:
                break

        # store results
        dist = self._distance(X, centroids)
        self.labels_ = np.argmin(dist, axis=1).astype(np.int32)
        self.centroids_ = centroids
        self.n_iter_ = step + 1

        # inertia = sum of (distance from each point to its own centroid)^2
        # a short way to say how "compact" the clusters are
        self.inertia_ = float(np.sum(np.min(dist, axis=1) ** 2))
        return self

    # -----------------------------------------------------------------
    def fit_predict(self, X):
        return self.fit(X).labels_
