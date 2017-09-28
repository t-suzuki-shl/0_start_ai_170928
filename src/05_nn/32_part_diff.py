import numpy as np

def my_func(x0, x1):
    y = 2 * x0**2 + x1**2
    return y

def part_diff(f, x0, x1):
    h = 1e-4
    y0 = (f(x0 + h, x1) - f(x0 - h, x1)) / (2 * h)
    y1 = (f(x0, x1 + h) - f(x0, x1 - h)) / (2 * h)
    return [y0, y1]

x0 = 2.0
x1 = 3.0
print(part_diff(my_func, x0, x1))
