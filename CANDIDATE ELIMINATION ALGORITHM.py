import pandas as pd
data = pd.read_csv(r"C:\Users\yuvak\OneDrive\Documents\python.txt")
features = data.columns[:-1]
most_specific_hypothesis = ['None'] * len(features)
most_general_hypothesis = ['?'] * len(features)
def update_specific_hypothesis(specific_h, instance):
    for i, value in enumerate(instance):
        if specific_h[i] == 'None':
            specific_h[i] = value
        elif specific_h[i] != value:
            specific_h[i] = '?'
    return specific_h
def update_general_hypothesis(general_h, instance):
    for i, value in enumerate(instance):
        if general_h[i] != '?' and general_h[i] != value:
            general_h[i] = '?'
    return general_h
for index, row in data.iterrows():
    instance, label = row.iloc[:-1], row.iloc[-1]
    if label == 'Yes':  
        most_specific_hypothesis = update_specific_hypothesis(most_specific_hypothesis, instance)
    elif label == 'No':  
        most_general_hypothesis = update_general_hypothesis(most_general_hypothesis, instance)
print("Most Specific Hypothesis:", most_specific_hypothesis)
print("Most General Hypothesis:", most_general_hypothesis)
