import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)

clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(x_train, y_train)

print(clf.score(x_test, y_test))

unknown_data = [[5.0, 4.1, 1.5, 0.5]]
print(clf.predict(unknown_data))
