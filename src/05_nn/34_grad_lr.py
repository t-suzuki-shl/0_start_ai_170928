import numpy as np

def my_func(x):
    y = 2 * x[0]**2 + x[1]**2
    return y

def grad(f, x):
    h = 1e-4
    y = np.zeros_like(x)
    for i in range(x.size):
        t = x[i]

        x[i] = t + h
        ya = f(x)
        x[i] = t - h
        yb = f(x)
        y[i] = (ya - yb) / (2 * h)

        x[i] = t
    return y

step = 10
lr = 0.1

x = np.array([2.0, 3.0])
for i in range(step):
    x -= lr * grad(my_func, x)
    print(x)
