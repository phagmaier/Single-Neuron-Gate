import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Inputs for OR gate
features = [[0, 0], [1, 0], [1, 1], [0, 1]]
#targets = [0, 1, 1, 1]
targets = [0,0,1,0]

def cost(weights, bias):
    result = 0.0
    for i in range(len(features)):
        x1, x2 = features[i]
        y = sigmoid(x1 * weights[0] + x2 * weights[1] + bias)
        d = y - targets[i]
        result += d * d
    result /= len(features)
    return result

def gradient(weights, bias):
    dw1, dw2, db = 0.0, 0.0, 0.0
    for i in range(len(features)):
        x1, x2 = features[i]
        y = sigmoid(x1 * weights[0] + x2 * weights[1] + bias)
        d = 2 * (y - targets[i]) * y * (1 - y)
        dw1 += d * x1
        dw2 += d * x2
        db += d
    dw1 /= len(features)
    dw2 /= len(features)
    db /= len(features)
    return dw1, dw2, db

def train(weights, bias, learning_rate, iterations):
    for i in range(iterations):
        dw1, dw2, db = gradient(weights, bias)
        weights[0] -= learning_rate * dw1
        weights[1] -= learning_rate * dw2
        bias -= learning_rate * db
        c = cost(weights, bias)
        print(f"The cost after iteration {i} is: {c}")

    results = [sigmoid(x1 * weights[0] + x2 * weights[1] + bias) for x1, x2 in features]
    print(f"The results after {iterations} iterations are:\n{results}")
    print(f"The actual results should be:\n{targets}")

weights = [0.5, 0.5]
bias = 0.5
learning_rate = 0.9
iterations = 1000

train(weights, bias, learning_rate, iterations)
