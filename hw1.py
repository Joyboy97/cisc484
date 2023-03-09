

prediction = x1*w1 + x2*w2 + x3*w3
data = [
    [2, 1, 3],
    [5, 2, 7],
    [2, -2, 4],
    [-3, 1, -2],
    [-5, 3, -1]
    ]

target_data = [6, 11, 0, 0, 1]
# TODO: put target data back into data
initial_weights = [.1, .2, .1, .1]
weights = initial_weights
learning_rate = .1
# o is from y equation plugging in weights from previous epoch
# y is calculated by plugging in 
error = (prediction - output)
w = wi + learning_rate(error)*x

output = 0

def sum_squared_error(data) -> int:
    squared_error = 0
    for row in data: 
        for y in target_data:
            prediction = row[0]*weights[1] + row[1]*weights[2] + row[2]*weights[3] + weights[0]
            squared_error += (prediction - y)
    return output




epochs_lapsed = 0
while (error != 0) or (epochs_lapsed < 10):




# go through each row in the dataset and get the err

# During one epoch:
# find sum squared error of the whole dataset.
# adjust each weight based on SSE, wi and xi. This is done 4 times because there are 4 weights.

# Carlson sample push
