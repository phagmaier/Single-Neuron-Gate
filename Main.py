import math
import numpy as np

#sig = lambda x: 1 / (1 + math.exp(-x))

'''
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
'''
# Inputs for OR/AND gate
features = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
targets = np.array([0, 1, 1, 1]).reshape(4, -1)
#targets = np.array([0, 0, 1, 0]).reshape(4, -1)



'''
Multiply weights to the features and add bias before taking the sigmoid of the result
vectorized sigmoid so it can be applied to all values in the matrix
returns the cost and the results cost is used whe taking the 'derivative' in the back function
always good to have results and we print them when done training 
'''
def forward(weights, bias, sig, calc, features=features, targets=targets):
    results = sig(features @ weights + bias)
    cost = (sum(calc(results, targets))) / len(results)
    return cost, results

'''
Performing our lazy version of 'backpropagation'
which really is just taking the finite difference
epsilon is the value we are adding on to the value f(x+h) - f(x) / h
eps functions as our h just a small adjustment to the function
we use this value to adjust our weight and bias i think theoretically we could also
do it to adjust our learning rate as well but I didn't do that.
Probbaly a more efficient way of doing this
'''
def backward(weights, c, eps, bias, sig, calc, lr):
    rows, cols = weights.shape
    new_weights = np.zeros_like(weights)
    for i in range(rows):
        for x in range(cols):
            temp = np.copy(weights)
            temp[i][x] += eps
            change = ((forward(temp, bias, sig, calc)[0])[0] - c) / eps
            new_weights[i][x] = weights[i][x] - (lr * change)
    change = ((forward(weights, bias+c, sig, calc)[0])[0] - c) / eps
    return new_weights, bias-lr*change



'''
Run our simple neural network (neuron) 
'''
def main(features=features, targets=targets, iterations=1000):
    # Set basic variables and vectorize basic lambda functions for vectors
    calc = np.vectorize(lambda x, y: (x - y) ** 2)
    sig = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))
    np.random.seed(13)
    weights = np.random.rand(2).reshape(2, 1)
    #weights = np.array([1,1]).reshape(2,1)
    bias = np.random.rand()
    lr = 0.1
    eps = 0.01

    for i in range(iterations):
        cost, results = forward(weights, bias, sig, calc)
        print(f"The cost after iteration {i} is: {cost}")
        weights,bias = backward(weights, cost, eps, bias, sig, calc, lr)

    print(f"The results after {iterations} iterations are:\n{results}")

    print(f'the actual results should be:\n {targets}')


main()

