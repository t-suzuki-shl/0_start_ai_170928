import numpy as np

def cross_entropy(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

y = np.array([0.1, 0.1, 0.8])
t = np.array([0, 0, 1])
print(cross_entropy(y, t))

y = np.array([0.1, 0.1, 0.8])
t = np.array([0, 1, 0])
print(cross_entropy(y, t))
