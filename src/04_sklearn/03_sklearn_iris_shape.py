import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
print(iris.keys())

print(iris.feature_names)
print(iris.data.shape)
print(iris.data[0])

print(iris.target_names)
print(iris.target.shape)
print(iris.target[0])
