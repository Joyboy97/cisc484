import numpy as np
import pandas as pd

data = {"y": [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
        "x1": [0, 1, 1, 0, 1, 0, 0, 0, 1, 0],
        "x2": [1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        "x3": [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
        "x4": [1, 0, 0, 1, 0, 1, 0, 0, 0, 1]}

df = pd.DataFrame(data)
print(df)
print("value counts of y")
print(df["y"].value_counts())
# calculate entropy
def entropy(column):
    value_counts = column.value_counts() 
    negative = 0
    positive = 0
    # if value is not 1 it's negative
    # count the number of positive and negative values
    # TODO: how do you access a series

print("main")
entropy(df["y"])
# calculate weighted entropy for each feature
# def weighted

# run ID3 algorithm on the dataframe
# def ID3():

