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