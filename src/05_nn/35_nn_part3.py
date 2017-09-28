import numpy as np

class NeuralNetwork:

    def __init__(self):
        self.hw = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        self.hb = np.array([0.1, 0.2])
        self.ow = np.array([[0.1, 0.2], [0.3, 0.4] ,[0.5, 0.6]])
        self.ob = np.array([0.1, 0.2, 0.3])

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
        result = np.zeros_like(x)
        for i, x in enumerate(x):
            result[i] = self.grad(f, x)
        return result

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


nn = NeuralNetwork()
x_train = np.array([[0.5, 0.6, 0.7], [0.6, 0.7, 0.8], [0.7, 0.8, 0.9]])
y_train = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

x_test = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
y_test = np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0]])

for i in range(x_train.shape[0]):
    nn.train(x_train[i], y_train[i])

correct = 0
for i in range(x_test.shape[0]):
    correct += nn.test(x_test[i], y_test[i])

print("Accuracy: {}".format(correct / x_test.shape[0]))
