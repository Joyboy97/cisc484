import numpy as np
import pandas as pd

# calculate entropy of a set
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

def count_pos_neg(df, feature: str, outcome_col):
    pos_mask = df.loc[(df[feature] == 1) & (df[outcome_col] == 1)]
    positives = pos_mask.shape[0]
    neg_mask = df.loc[(df[feature] == 1) & (df[outcome_col] != 1)]
    negatives = neg_mask.shape[0]
    total = positives + negatives
    return positives, negatives, total

def information_gain(df, outcome_col: str, feature: str):
    # calculate entropy for feature
    # calculate weighted entropy for each feature
    set_entropy = entropy_of_set(df[outcome_col])
    positives, negatives, total = count_pos_neg(df, feature, outcome_col)
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

def max_info_gain(df, outcome_col: str):
    # find the feature with the highest information gain
    print("information gain for each feature: ")
    info_gain_dict = {}
    for feature in df.columns:
        if feature != outcome_col:
            info_gain_dict.update({feature: information_gain(df, outcome_col, feature)})
    # select the feature with the highest information gain
    chosen_feature = max(info_gain_dict, key=info_gain_dict.get)
    print(info_gain_dict)
    print(chosen_feature)
    return chosen_feature

def check_pure(positives, negatives, total):
    if positives == 0 or negatives == 0:
        return True
    else:
        return False

# run ID3 algorithm on the dataframe
def iterative_ID3(df, outcome_col: str):
    tree = Decision_tree(None)
    for i in range(0, 2):
        if tree.root == None:
            print("root:")
            # find the feature with the highest information gain
            chosen_feature = max_info_gain(df, outcome_col)
            positives, negatives, total = count_pos_neg(df, chosen_feature, outcome_col)
            print(positives, negatives, total)
            node = Node(chosen_feature, {}, None, True)
            if check_pure(positives, negatives, total):
                print("pure")
                # what's the other node become? maybe it gets set in the next call/iteration
                if positives > negatives:
                    node.add_attribute_child(1, "positive")
                else:
                    node.add_attribute_child(1, "negative")
            node = Node(chosen_feature, {0: None, 1: None}, None, True)
            tree.root = node
            # remove the chosen feature from the dataframe
            df = df.drop(columns=[chosen_feature])
            print(df)
        else: 
            print("not root")
            chosen_feature = max_info_gain(df, outcome_col)
            positives, negatives, total = count_pos_neg(df, chosen_feature, outcome_col)
            print(positives, negatives, total)
            # node = Node(chosen_feature, {0: None, 1: None}, None, True)
            # remove the chosen feature from the dataframe
            # df = df.drop(columns=[chosen_feature])
            print(df)
    # populate
    return

def ID3(df, outcome_col: str, tree):
    # recursive ID3
    chosen_feature = max_info_gain(df, outcome_col)

    # basecase: if a choice (0 or 1) is pure, return which choice is pure and what the outcome is
    positives, negatives, total = count_pos_neg(df, outcome_col, chosen_feature)
    if check_pure(positives, negatives, total):
        if positives > negatives:
            return "positive"
        else:
            return "negative"

    # basecase: if there are no more features to choose from, return the most probable outcome
    # df = df.drop(columns=["x2", "x3", "x4"])
    if df.shape[1] == 2: # if there are no more features to choose from
        # return the most probable outcome
        # remaining_feature = df.columns[df.columns != outcome_col][0]
        # print(remaining_feature)
        # positive, negative, total = count_pos_neg(df, outcome_col, remaining_feature)
        if positive > negative:
            return "positive"
        else: 
            return "negative"
    

    # general case: 
    # 1. find the feature with the highest information gain
    # 2. initialize a node of that feature and add it to the tree in the correct place
    # 3. split the dataframe into two dataframes based on the feature's value
    # 4. run ID3 on each of the two dataframes
    
    # 0s dataframe:
    zeros_df = df.loc[df[chosen_feature] == 0]
    # 1s dataframe:
    ones_df = df.loc[df[chosen_feature] == 1]


    if tree.root == None:
        tree = Decision_tree(Node(chosen_feature, {0: ID3(zeros_df, outcome_col, tree), 
                                                    1: ID3(ones_df, outcome_col, tree)}, None, True))
        return tree

    else: 
        tree.root.add_attribute_child(0, ID3(zeros_df, outcome_col, tree))
        tree.root.add_attribute_child(1, ID3(ones_df, outcome_col, tree))


class Node:
    # def __init__(self, feature: str, attribute_to_child: dict, level: int, parent, root_bool: bool = False):
    def __init__(self, feature: str, attribute_to_child: dict, parent, root_bool: bool = False):
        self.feature = feature
        self.attribute_to_child = attribute_to_child
        self.parent = parent
        # self.level = level
        self.root_bool = root_bool
    
    def print_node(self, level=0):
        to_return = "\t"*level + repr(self.feature) + "\n"
        for attribute in self.attribute_to_child:
            if type(self.attribute_to_child[attribute]) == str:
                to_return += "\t"*(level+1) + repr(attribute) + " -> " + repr(self.attribute_to_child[attribute]) + "\n"
            else: 
                to_return += self.attribute_to_child[attribute].print_node(level+1)
        return to_return

        # else: 
        #     to_return += 
        # return self.feature + 
    
    def __repr__(self):
        return "treenode representation"
    
    def add_attribute_child(self, attribute: object, child):
        self.attribute_to_child.update({attribute: child})
        return
    

class Decision_tree:
    def __init__(self, root: Node):
        self.root = root
    
    # print the tree
    def __str__(self):
        return self.root.print_node()
        
def main():
    data = {"y": [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
            "x1": [0, 1, 1, 0, 1, 0, 0, 0, 1, 0],
            "x2": [1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            "x3": [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
            "x4": [1, 0, 0, 1, 0, 1, 0, 0, 0, 1]}
    df = pd.DataFrame(data)
    print(df)
    print(ID3(df, "y", Decision_tree(None)))


    # node1 = Node("x2", {0: "x4", 1: "x4"}, None, False)
    # node = Node("x1", {0: node1, 1: "x3"}, None, True)
    # print(node.print_node())
    return

if __name__ == "__main__":
    main()