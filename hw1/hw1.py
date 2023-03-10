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

sum_squared_error(data, weights):w0