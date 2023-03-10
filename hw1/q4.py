data = [
    [-1, -2, -3, -4],
    [1, 2, 3, 4]
    ]

target_data = [[0, 0, 0, 0], [1, 1, 1, 1]]
initial_weights = [.1, .1]
weights = initial_weights
print("initial weights:")
print(initial_weights)
learning_rate = .01
epoch = 20

def sum_squared_error(data, weights) -> int:
    squared_error = 0
    for row in data: 
        for k in target_data:
            for i in range(4):
                prediction = row[i]*k[i]+ weights[0]
                squared_error += (prediction - row[i])**2
            # print(squared_error)
    print ("Squared Error:", squared_error, "---------------")
    return squared_error

# multiply x's by their errors then add them all up and multiply by learning rate
# prediction * x
# output is not individualized its just the prediction of the whole thing
# w = wi + learning_rate((xi)(yi - prediction))
while epoch > 0:
    sum_squared_error(data, weights)
    delta_w = [0, 0,]
    for row in data:
        for k in target_data:
            for i in range(4):
                prediction = row[i]*k[i] + weights[0]
                y = k[i]
                error = y - prediction # error for each row
        for i in range(len(data)):
            delta_w[i] += error*row[i]
    delta_w = [x*learning_rate for x in delta_w]

    for i in range(len(weights)):
        weights[i] += delta_w[i]
    print("updated weights:")
    print(weights)
    print("epoch: " + str(epoch))
    epoch -= 1

sum_squared_error(data, weights)