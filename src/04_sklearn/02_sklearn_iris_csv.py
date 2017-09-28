import numpy as np
from sklearn.neighbors import KNeighborsClassifier

train = np.loadtxt("train01.csv",delimiter=",")
test = np.loadtxt("test01.csv",delimiter=",")

x_train = train[:,1:]
y_train = train[:,0]

x_test = test[:,1:]
y_test = test[:,0]

clf = KNeighborsClassifier()
clf.fit(x_train, y_train)

print(clf.predict(x_test))
print(y_test)
print(clf.score(x_test, y_test))
