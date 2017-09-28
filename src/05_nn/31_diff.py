import numpy as np

def diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

def my_func(x):
    y = 2 * x**2 + x + 1
    return y

print(diff(my_func, 1.0))
print(diff(my_func, 2.0))
