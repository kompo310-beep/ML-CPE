## Simple KNN Classifier implemented with NumPy (for beginners)
## Same class name/API as the TensorFlow version, so main.py needs no changes.

import numpy as np


class TFKNNClassifier:

    def __init__(self, k=5):
        self.k = k          # number of neighbors to use

    # -----------------------------------------------------------------
    def fit(self, X, y):    # Train KNN (just store the data)

        self.X_train = np.asarray(X, dtype=np.float32)
        self.y_train = np.asarray(y, dtype=np.int32)
        self.n_classes = int(y.max()) + 1
        return self

    # -----------------------------------------------------------------
    def _distance(self, X_new):
        """
        Euclidean distance = sqrt( (x1-y1)^2 + (x2-y2)^2 + ... )
        """
        diff = X_new[:, None, :] - self.X_train[None, :, :]     # pairwise differences
        return np.sqrt(np.sum(diff ** 2, axis=2))                # (n_new, n_train)

    # -----------------------------------------------------------------
    def predict(self, X):
        """predict class of new data and return array of class labels"""
        X = np.asarray(X, dtype=np.float32)

        # step 1 : distance
        dist = self._distance(X)

        # step 2 : select k nearest
        # argsort sorts ascending, so the first k columns are the closest
        idx = np.argsort(dist, axis=1)[:, :self.k]        # idx = positions of the neighbors
        neighbor_labels = self.y_train[idx]                # classes of the neighbors (n_new, k)

        # step 3 : vote among the k neighbors
        votes = np.zeros((X.shape[0], self.n_classes), dtype=np.int32)
        for c in range(self.n_classes):
            votes[:, c] = np.sum(neighbor_labels == c, axis=1)

        return np.argmax(votes, axis=1)                    # class with the most votes

    # -----------------------------------------------------------------
    def score(self, X, y):
        """calculate accuracy = proportion of correct predictions"""
        return float(np.mean(self.predict(X) == y))
