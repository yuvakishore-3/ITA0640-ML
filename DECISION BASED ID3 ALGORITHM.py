import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from sklearn import tree
import matplotlib.pyplot as plt
playTennis = pd.read_csv(r"C:\Users\yuvak\OneDrive\Documents\data.txt")
print(playTennis.columns)
playTennis.rename(columns={
    'Outlook': 'outlook',
    'Temperature': 'temp',
    'Humidity': 'humidity',
    'Wind': 'wind',
    'PlayTennis': 'play'
}, inplace=True)
print(playTennis.columns)
Le = LabelEncoder()
playTennis['outlook'] = Le.fit_transform(playTennis['outlook'])
playTennis['temp'] = Le.fit_transform(playTennis['temp'])
playTennis['humidity'] = Le.fit_transform(playTennis['humidity'])
playTennis['wind'] = Le.fit_transform(playTennis['wind'])
playTennis['play'] = Le.fit_transform(playTennis['play'])
Y = playTennis['play']
X = playTennis.drop(['play'], axis=1)
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, Y)
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.show()
