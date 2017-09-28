import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

iris = load_iris()

model = KMeans(n_clusters=3)

model.fit(iris.data)

print(model.labels_)

plt.scatter(iris.data[model.labels_==0,0], iris.data[model.labels_==0,1], marker='o', color='red', label='cluster 0')
plt.scatter(iris.data[model.labels_==1,0], iris.data[model.labels_==1,1], marker='x', color='green', label='cluster 1')
plt.scatter(iris.data[model.labels_==2,0], iris.data[model.labels_==2,1], marker='^', color='blue', label='cluster 2')

plt.legend()
plt.show()
