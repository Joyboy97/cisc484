x = [[1,2,1,3], [1,5,2,7], [1,2,-2,4], [1,-3,1,-2], [1,-5,3,-1]]
y = [6,11,0,0,1]
w = [0.1, 0.2, 0.1, 0.1]

learning_rate = 0.01
epochs = 10

def dot(x,y):
    return sum(i*j for i, j in zip(x, y))

def dot2(w, x):
    return sum(i*j for i, j in zip(w, x))


def compute_cost(x, y, w):
    return sum((y-dot(x, w))**2)

def gradient_descent(x, y, w, learning_rate, epochs):
    delta_w = w
    print(f"Initial Cost: {compute_cost(x, y, w)}")

    for i in range(epochs):
        for j in range(len(w)):
            accum_sum = 0
            for k in range(len(x)):
                y_pred = dot2(w, x[k])
                accum_sum += (y_pred - y[k]*x[k][j])
            delta_w[j] = learning_rate *accum_sum
        w = w - delta_w
        print(f"Epoch {e} Cost: {compute_cost(x, y, w)}")

final_weights = gradient_descent(x, y, w, learning_rate, epochs)