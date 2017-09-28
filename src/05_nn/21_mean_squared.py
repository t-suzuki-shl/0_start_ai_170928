import numpy as np

def mean_squared(y, t):
    return 0.5 * np.sum((y - t)**2)

y = np.array([0.1, 0.1, 0.8])
t = np.array([0, 0, 1])
print(mean_squared(y, t))

y = np.array([0.1, 0.1, 0.8])
t = np.array([0, 1, 0])
print(mean_squared(y, t))
