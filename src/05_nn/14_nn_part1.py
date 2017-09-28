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

    def input(self, x):
        hy = self.neuron(self.hw, self.hb, x, self.sigmoid)
        y = self.neuron(self.ow, self.ob, hy, self.softmax)
        return y


nn = NeuralNetwork()
x = np.array([0.5, 0.6, 0.7])
print(nn.input(x))
