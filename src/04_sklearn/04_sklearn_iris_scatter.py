import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
data = np.c_[iris.target, iris.data]

plt.scatter(data[data[:,0]==0,1], data[data[:,0]==0,2], marker='o', color='red', label='setosa')
plt.scatter(data[data[:,0]==1,1], data[data[:,0]==1,2], marker='x', color='green', label='versicolor')
plt.scatter(data[data[:,0]==2,1], data[data[:,0]==2,2], marker='^', color='blue', label='virginica')

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.legend()

plt.show()
