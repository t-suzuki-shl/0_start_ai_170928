import numpy as np

def neuron(x):
    w = np.array([0.8, -0.2])
    b = -0.5
    y = w.dot(x) + b
    if y > 0 :
        return 1
    else :
        return 0

print(neuron(np.array([1, 0])))
print(neuron(np.array([0, 1])))
