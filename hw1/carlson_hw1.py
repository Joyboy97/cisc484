# Q3
data = [
    [2, 1, 3, 6],
    [5, 2, 7, 11],
    [2, -2, 4, 0],
    [-3, 1, -2, 0],
    [-5, 3, -1, 1]
    ]

# w1, w2, w3, w0
weights = [.2, .1, .1, .1] # w0 is on the end as .1
learning_rate = .01

def sum_squared_error(row):
    # sum squared error of one row
    y = row[3]
    return (y - (row[0]*weights[1] + row[1]*weights[2] + row[2]*weights[3] + weights[0]))**2

# def linear_regression(data, weights):
#     error = 1
#     # while (error != 0) and (epochs_lapsed < 10):
#         # calculate x 
#         summed_x = [0, 0, 0, 1]
#         for row in data:
#             for i in range(len(row) - 1):
#                 summed_x[i] += row[i]
#         print("summed x: ", summed_x)
#         # calculate SSE
#         error = 0
#         for row in data: 
#             error += sum_squared_error(row)
#             print(error)
#         print("final error for this epoch: ", error)
#         # calculate change in weights

def linear_regression(data, weights):
    summed_x = [0, 0, 0, 1]
    for row in data:
        for i in range(len(row) - 1):
            summed_x[i] += row[i]
    print("summed x: ", summed_x)
    # calculate SSE
    error = 0
    for row in data: 
        error += sum_squared_error(row)
        print(error)
    print("final error for this epoch: ", error)

linear_regression(data, weights)