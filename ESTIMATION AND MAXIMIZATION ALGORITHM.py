import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
gmm = GaussianMixture(n_components=3, random_state=0)
gmm.fit(X)
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
centers = gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('Gaussian Mixture Model (EM Algorithm)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
print("Cluster Centers:\n", centers)
print("Covariances:\n", gmm.covariances_)
print("Weights:\n", gmm.weights_)
