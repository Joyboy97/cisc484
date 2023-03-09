import numpy as np
X_train = np.array([[1, 2, 1, 3], [1, 5, 2, 7], [1, 2, -2, 4], [1, -3, 1, -2], [1, -5, 3, -1]])
y_train = np.array([6, 11, 0, 0, 1])

# Define the weights
w = np.array([0.1, 0.2, 0.1, 0.1])


def compute_cost(X, y, w):
    return np.sum((y - np.dot(X, w))**2)

def gradient_descent(X, y, w, learning_rate, epochs):
    delta_w = np.empty_like(w)

    print(f"Initial Cost: {compute_cost(X, y, w)}")
    
    for e in range(epochs):

        # calculate delta_w
        for i in range(len(w)):
            accum_sum = 0
            for t in range(len(X)):
                y_pred = np.dot(w, X[t])
                accum_sum += (y_pred - y[t]) * X[t][i]

            # print(accum_sum)
            delta_w[i] = learning_rate * accum_sum

        w = w - delta_w

        print(f"Epoch {e} Cost: {compute_cost(X, y, w)}")
    return w

learning_rate = 0.01
epochs = 10

final_weights = gradient_descent(X_train, y_train, w, learning_rate, epochs)