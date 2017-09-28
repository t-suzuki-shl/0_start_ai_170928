from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 'setosa', 'versicolor', 'virginica'
iris = load_iris()

# # clf 'versicolor' / 'virginica'

# # train.csv
# petals = iris.data[50:80,2:]
# for petal in petals:
#     print("1,{},{}".format(petal[0], petal[1]))
#
# petals = iris.data[100:130,2:]
# for petal in petals:
#     print("2,{},{}".format(petal[0], petal[1]))

# # test.csv
petals = iris.data[80:100,2:]
for petal in petals:
    print("1,{},{}".format(petal[0], petal[1]))

petals = iris.data[130:150,2:]
for petal in petals:
    print("2,{},{}".format(petal[0], petal[1]))
