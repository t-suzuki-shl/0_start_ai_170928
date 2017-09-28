import numpy as np

class NeuralNetwork:

    def __init__(self):
        self.hw = 0.01 * np.random.randn(50, 4)
        self.hb = np.zeros(50)
        self.ow = 0.01 * np.random.randn(3, 50)
        self.ob = np.zeros(3)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def neuron(self, w, b, x, activate):
        return activate(w.dot(x) + b)

    def cross_entropy(self, y, t):
        delta = 1e-7
        return -np.sum(t * np.log(y + delta))

    def input(self, x, t):
        hy = self.neuron(self.hw, self.hb, x, self.sigmoid)
        y = self.neuron(self.ow, self.ob, hy, self.softmax)
        loss = self.cross_entropy(y, t)
        return loss

    def grad(self, f, x):
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

    def grad_2dim(self, f, x):
        grad = np.zeros_like(x)
        for i, x in enumerate(x):
            grad[i] = self.grad(f, x)
        return grad

    def train(self, x, t, lr=0.1):
        loss = lambda n: self.input(x, t)

        grads = {}
        grads['hw'] = self.grad_2dim(loss, self.hw)
        grads['hb'] = self.grad(loss, self.hb)
        grads['ow'] = self.grad_2dim(loss, self.ow)
        grads['ob'] = self.grad(loss, self.ob)

        self.hw -= lr * grads['hw']
        self.hb -= lr * grads['hb']
        self.ow -= lr * grads['ow']
        self.ob -= lr * grads['ob']

    def test(self, x, t):
        hy = self.neuron(self.hw, self.hb, x, self.sigmoid)
        y = self.neuron(self.ow, self.ob, hy, self.softmax)
        return (np.argmax(y) == np.argmax(t)).astype('int')

    def to_categorical(self, y, num_classes):
        y = np.array(y, dtype='int').ravel()
        n = y.shape[0]
        categorical = np.zeros((n, num_classes))
        categorical[np.arange(n), y] = 1
        return categorical


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

nn = NeuralNetwork()

iris = load_iris()
x = iris.data
y = nn.to_categorical(iris.target, 3)
x_train, x_test, y_train, y_test = train_test_split(x, y)

for epoch in range(5):
    print(epoch)
    for i in range(x_train.shape[0]):
        nn.train(x_train[i], y_train[i])

correct = 0
for i in range(x_test.shape[0]):
    correct += nn.test(x_test[i], y_test[i])
print(correct / x_test.shape[0])
