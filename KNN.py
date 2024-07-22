import numpy as np
from collections import Counter
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))
def knn_predict(X_train, y_train, X_test, k=3):
    predictions = []
    for test_point in X_test:
        distances = []
        for i, train_point in enumerate(X_train):
            distance = euclidean_distance(test_point, train_point)
            distances.append((distance, y_train[i]))
        distances.sort(key=lambda x: x[0])
        k_nearest_neighbors = [dist[1] for dist in distances[:k]]
        most_common = Counter(k_nearest_neighbors).most_common(1)
        predictions.append(most_common[0][0])
    return np.array(predictions)
if __name__ == "__main__":
    X_train = np.array([[1, 2], [2, 3], [3, 4], [6, 7]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[2, 2], [7, 7]])
    predictions = knn_predict(X_train, y_train, X_test, k=3)
    print("Predictions:", predictions)
