import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
np.random.seed(42)
n_samples = 2000
n_features = 10
X = np.random.rand(n_samples, n_features) * 100
y = np.random.randint(0, 4, n_samples)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier()
}
for name, clf in classifiers.items():
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f"{name}: Mean accuracy = {np.mean(scores):.2f}, Std = {np.std(scores):.2f}")
best_clf = RandomForestClassifier()
best_clf.fit(X_train, y_train)
predictions = best_clf.predict(X_test)
print("Sample predictions:", predictions[:10])
