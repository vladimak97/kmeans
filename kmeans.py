
# Implement the K-Means clustering algorithm in Python.

import numpy as np

def kmeans_clustering(X, k, max_iterations=100):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    for _ in range(max_iterations):
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

X = np.array([[1, 2], [2, 3], [5, 6], [6, 7], [9, 8], [10, 9]])
k = 2
labels, centroids = kmeans_clustering(X, k)
print(f"Cluster Labels: {labels}")
print(f"Cluster Centroids: {centroids}")