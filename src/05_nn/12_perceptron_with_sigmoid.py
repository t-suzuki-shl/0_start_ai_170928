import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def neuron(x):
    w = np.array([0.8, -0.2])
    b = -0.5
    y = sigmoid(w.dot(x) + b)
    return y

print(neuron(np.array([1, 0])))
print(neuron(np.array([0, 1])))
