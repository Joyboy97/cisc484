# Q3
data = [
    [2, 1, 3, 6],
    [5, 2, 7, 11],
    [2, -2, 4, 0],
    [-3, 1, -2, 0],
    [-5, 3, -1, 1]
    ]

# target_data = [6, 11, 0, 0, 1]
initial_weights = [.2, .1, .1, .1] # w0 is on the end as .1
weights = initial_weights
print("initial weights:")
print(initial_weights)
learning_rate = .01
epoch = 10

def sum_squared_error(data, weights) -> int:
    squared_error = 0
    for row in data: 
        prediction = row[0]*weights[0] + row[1]*weights[1] + row[2]*weights[2] + weights[3]
        squared_error += (prediction - row[3])**2
        # print(squared_error)
    print ("Squared Error:", squared_error, "---------------")
    return squared_error

# multiply x's by their errors then add them all up and multiply by learning rate
# prediction * x
# output is not individualized its just the prediction of the whole thing
# w = wi + learning_rate((xi)(yi - prediction))
while epoch > 0:
    sum_squared_error(data, weights)
    delta_w = [0, 0, 0, 0]
    for row in data: 
        prediction = row[0]*weights[0] + row[1]*weights[1] + row[2]*weights[2] + weights[3]
        y = row[3]
        error = y - prediction # error for each row
        for i in range(len(row) - 1):
            delta_w[i] += error*row[i]
        delta_w[3] += error*1
    delta_w = [x*learning_rate for x in delta_w]

    for i in range(len(weights)):
        weights[i] += delta_w[i]
    print("updated weights:")
    print(weights)
    print("epoch: " + str(epoch))
    epoch -= 1

sum_squared_error(data, weights)

import numpy as np
import matplotlib.pyplot as plt
final_predictions=[]
X_train = np.array([
    [-1, -2, -3, -4],
    [1, 2, 3, 4]
])

data = [
    [-1, -2, -3, -4],
    [1, 2, 3, 4]
]

target_data = [0, 1]
final_predictions = [[0, 0, 0, 0], [1, 1, 1, 1]]

initial_weights = [.1, .1]
weights = initial_weights
print("initial weights:")
print(initial_weights)
learning_rate = .01
epoch = 0


def sum_squared_error(data, weights) -> int:
    squared_error = 0
    for j in range(2):
        for i in range(4):
            prediction = data[j][i] * weights[1] + weights[0]
        squared_error += (prediction - target_data[j])**2
        # print(squared_error)
        # final_predictions.append(prediction)
    print("Squared Error:", squared_error, "---------------")
    return squared_error


# multiply x's by their errors then add them all up and multiply by learning rate
# prediction * x
# output is not individualized its just the prediction of the whole thing
# w = wi + learning_rate((xi)(yi - prediction))
while epoch < 20:
    sum_squared_error(data, weights)

    delta_w = [0, 0]
    for j in range(len(data)):
        # for k in target_data:
        for i in range(4):
            prediction = data[j][i]*weights[1] + weights[0]
        y = target_data[j]
        error = y - prediction
        # print(f"{error}-------------------error")  # error for each row
        for i in range(4):
            delta_w[1] += error*data[j][i]
        delta_w[0] += error*1

    delta_w = [x*learning_rate for x in delta_w]

    for i in range(len(weights)):
        weights[i] += delta_w[i]
    print("updated weights:")
    print(weights)
    print("epoch: " + str(epoch))
    epoch += 1

sum_squared_error(data, weights)
print(X_train)
# l = mlines.Line2D([xmin,xmax], [ymin,ymax])
# ax.add_line(l)
plt.scatter(X_train, final_predictions, label='induced model')
# plt.scatterplot(X_train,  final_predictions, label='induced model')
# training_points = [[0, 0],[0, 0],[0, 0],[0, 0],[1,1], [1,1], [1,1], [1,1]]
plt.show()

import pickle
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Q5.1
# DiabetesPedigreeFuncition has the largest impact on the prediction
print("Q5.1 -------------------------")
train_df = pd.read_pickle('./HW1_Q5/train_1.pkl')
train_x = train_df.loc[:, train_df.columns != 'Outcome']
train_y = train_df.loc[:,'Outcome']
logisticRegr = LogisticRegression(max_iter=100000)
logisticRegr.fit(train_x, train_y)

test_df = pd.read_pickle('./HW1_Q5/test_1.pkl')
test_x = test_df.loc[:, test_df.columns != 'Outcome']
test_y = test_df.loc[:,'Outcome']
predictions = logisticRegr.predict(test_x)
print(accuracy_score(test_y, predictions))
pred_df = pd.DataFrame(predictions, columns=['Prediction'])
pred_df_with_context = pd.merge(test_x, pred_df, left_index=True, right_index=True)
print(pred_df_with_context)
weights = pd.DataFrame(logisticRegr.coef_, columns=train_x.columns)
print(weights)

print("Q5.2-------------------------")
train_df2 = pd.read_pickle('./HW1_Q5/train_2.pkl')
train_x2 = train_df2.loc[:, train_df2.columns != 'Outcome']
train_y2 = train_df2.loc[:,'Outcome']
logisticRegr2 = LogisticRegression(max_iter=100000)
logisticRegr2.fit(train_x2, train_y2)

test_df2 = pd.read_pickle('./HW1_Q5/test_2.pkl')
test_x2 = test_df2.loc[:, test_df2.columns != 'Outcome']
test_y2 = test_df2.loc[:,'Outcome']
predictions2 = logisticRegr2.predict(test_x2)
print(accuracy_score(test_y2, predictions2))
pred_df2 = pd.DataFrame(predictions2, columns=['Prediction'])
pred_df_with_context2 = pd.merge(test_x2, pred_df2, left_index=True, right_index=True)
print(pred_df_with_context2)
weights2 = pd.DataFrame(logisticRegr2.coef_, columns=train_x2.columns)
print(weights2)

# The new feature's weight is quite low at .016896, same as glucose's. This makes sense
# because the new feature is identical to Glucose.
# The values are the same because the new feature is identical to Glucose.

print("Q5.3-------------------------")
train_df3 = pd.read_pickle('./HW1_Q5/train_3.pkl')
train_x3 = train_df3.loc[:, train_df3.columns != 'Outcome']
train_y3 = train_df3.loc[:,'Outcome']
logisticRegr3 = LogisticRegression(max_iter=100000)
logisticRegr3.fit(train_x3, train_y3)

test_df3 = pd.read_pickle('./HW1_Q5/test_3.pkl')
test_x3 = test_df3.loc[:, test_df3.columns != 'Outcome']
test_y3 = test_df3.loc[:,'Outcome']
predictions3 = logisticRegr3.predict(test_x3)
print(accuracy_score(test_y3, predictions3))
pred_df3 = pd.DataFrame(predictions3, columns=['Prediction'])
pred_df_with_context3 = pd.merge(test_x3, pred_df3, left_index=True, right_index=True)
print(pred_df_with_context3)
weights3 = pd.DataFrame(logisticRegr3.coef_, columns=train_x3.columns)
print(weights3)