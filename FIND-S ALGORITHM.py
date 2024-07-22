import pandas as pd
data = pd.read_csv(r"C:\Users\yuvak\OneDrive\Documents\data.txt")
def initialize_hypothesis(num_attributes):
    return ['0'] * num_attributes
def find_s_algorithm(data):
    num_attributes = len(data.columns) - 1
    hypothesis = initialize_hypothesis(num_attributes)
    for index, row in data.iterrows():
        if row.iloc[-1].strip().lower() == 'yes':  
            if hypothesis == ['0'] * num_attributes:
                hypothesis = row.iloc[:-1].tolist()
            else:
                for i in range(num_attributes):
                    if hypothesis[i] != row.iloc[i]:
                        hypothesis[i] = '?'
    return hypothesis
hypothesis = find_s_algorithm(data)
print("The most specific hypothesis is:", hypothesis)
