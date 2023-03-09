# prediction = x1*w1 + x2*w2 + x3*w3

# added one to the front for w0 since w0 is always one
data = [
    [1, 2, 1, 3],
    [1, 5, 2, 7],
    [1, 2, -2, 4],
    [1, -3, 1, -2],
    [1, -5, 3, -1]
    ]

target_data = [6, 11, 0, 0, 1]
initial_weights = [.1, .2, .1, .1] 
weights = initial_weights
learning_rate = .01
epochs = 10
# o is from y equation plugging in weights from previous epoch
# y is calculated by plugging in 
# error = (prediction - output)
# w = wi + learning_rate(error)*x

# def sum_squared_error(data, weights) -> int:
#     squared_error = 0
#     for row in data: 
#         prediction = row[0]*weights[1] + row[1]*weights[2] + row[2]*weights[3] + weights[0]
#         squared_error += (prediction - row[3])**2
#         print(squared_error)
#     # print (squared_error)
#     return squared_error

def adjust_weight(data, weights) -> int:
    squared_error = 0
    for row in data: 
        prediction = row[0]*weights[1] + row[1]*weights[2] + row[2]*weights[3] + weights[0]
        squared_error += (row[3] - prediction)**2
        print(squared_error)
    # 1 epoch
    for row in data:
        for i in range(len(row) - 1):
            weights[i] = weights[i] + learning_rate*squared_error*row[i]
        weights[3] = weights[3] + learning_rate*squared_error*1


# epochs_lapsed = 0
# while (error != 0) or (epochs_lapsed < 10):
#     for weight in weights:
#         weight = weight + learning_rate*sum_squared_error(data, weights)
# sum_squared_error(data, initial_weights)


# go through each row in the dataset and get the err

# During one epoch:
# find sum squared error of the whole dataset.
# adjust each weight based on SSE, wi and xi. This is done 4 times because there are 4 weights.
# the problem is i need only 4 xi's. 
# add up all xis together, equivalent to transposing x?
# Carlson sample push
