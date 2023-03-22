import numpy as np
import pandas as pd

data = {"y": [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
        "x1": [0, 1, 1, 0, 1, 0, 0, 0, 1, 0],
        "x2": [1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        "x3": [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
        "x4": [1, 0, 0, 1, 0, 1, 0, 0, 0, 1]}

df = pd.DataFrame(data)
print(df)
# calculate entropy
def entropy_of_set(column):
    value_counts = column.value_counts() 
    negative = 0
    positive = 0
    # if value is not 1 it's negative
    # count the number of positive and negative values
    # print(value_counts)
    for i in value_counts.index:
        if i == 1:
            positive = value_counts.loc[i]
        else:
            negative = value_counts.loc[i]
    # print("positive: " + str(positive))
    # print("negative: " + str(negative))
    total = positive + negative
    entropy = -((positive/total) * np.log2(positive/total)) - ((negative/total) * np.log2(negative/total))
    return entropy

def information_gain(df, outcome_col: str, feature: str):
    # calculate entropy for feature
    # calculate weighted entropy for each feature
    set_entropy = entropy_of_set(df[outcome_col])
    # find all the 1's that also have a 1 in the outcome column
    # find all the 1's that also have a -1 in the outcome column
    pos_mask = df.loc[(df[feature] == 1) & (df[outcome_col] == 1)]
    positives = pos_mask.shape[0]
    neg_mask = df.loc[(df[feature] == 1) & (df[outcome_col] != 1)]
    negatives = neg_mask.shape[0]
    total = positives + negatives
    # print("entropy of set: " + str(set_entropy))
    # print("positives: " + str(positives))
    # print("negatives: " + str(negatives))
    # print("total: " + str(total))
    # print(df.shape[0])
    if positives == 0 or negatives == 0:
        information_gain = set_entropy
    else:
        information_gain = set_entropy - ((total/df.shape[0]) * (
            -(positives/total) * np.log2(positives/total) - (negatives/total) * np.log2(negatives/total)))
    return information_gain

print("main")
print(information_gain(df, "y", "x1"))
print(information_gain(df, "y", "x2"))
print(information_gain(df, "y", "x3"))
print(information_gain(df, "y", "x4"))

# calculate weighted entropy for each feature
# def weighted

# run ID3 algorithm on the dataframe
# def ID3():