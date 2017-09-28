## 3.9 irisデータによる検証

具体的なデータを使って検証してみましょう。ここではscikit-learnに付属するiris（アヤメ）データを使ってニューラルネットワークの学習の検証します。

<img src="img/03_13.png?ab" width="500px">

ニューラルネットワークの入力層は4、隠れ層は50、出力層は3で構成されるものとします。

<div style="page-break-before:always"></div>

```python
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
```

プログラムの実行結果は次のようになるでしょう。

```
0
1
2
3
4
0.947368421053
```

> 実際には実行の都度結果は変化するでしょう。概ね60%〜90%の正答率を期待できるでしょう。

<div style="page-break-before:always"></div>

さて上記のコードにはいくつか変更があります。

まずはニューラルネットワークの重みとバイアスの初期値です。

```python
def __init__(self):
    self.hw = 0.01 * np.random.randn(50, 4)
    self.hb = np.zeros(50)
    self.ow = 0.01 * np.random.randn(3, 50)
    self.ob = np.zeros(3)
```

重みの初期値は乱数に、バイアスは0で初期化しています。また入力層のニューロン数はiridデータのカラム数に合わせて 4 としています。次に隠し層のニューロン数は 50 としています。隠し層のニューロン数は10でも30でも構いません。出力層のニューロン数は 3 としています。これはirisデータは3種類（0:setosa, 1:versicolor, 2:virginica）に分類されるからです。

> 隠し層のニューロンの数が多いとニューラルネットワークの表現力は高まりますが、多すぎると処理に時間がかかるだけでなく過学習を引き起こす原因にもなります。

それからNeuralNetworkクラスにto_categoricalメソッドを実装しています。

```python
def to_categorical(self, y, num_classes):
    y = np.array(y, dtype='int').ravel()
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical
```

これはニューラルネットワークの本質とは関係のないユーティリティメソッドです。教師データの期待値をone-hotラベル表現に変換します。

> one-hotラベルとは、0の場合 [1, 0, 0] 1の場合 [0, 1, 0] 2の場合[0, 0, 1] のようにスカラー値を配列に変換します。

<div style="page-break-before:always"></div>


irisデータをロードする部分は次のようになります。

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

nn = NeuralNetwork()

iris = load_iris()
x = iris.data
y = nn.to_categorical(iris.target, 3)
x_train, x_test, y_train, y_test = train_test_split(x, y)
```

scikit-learnに含まれるtrain_test_splitを使って、訓練データとテストデータに分割しています。また期待値を先のto_categoricalメソッドを使ってone-hotラベル表現に変更しています。


最後に訓練データを与える部分を見てみましょう。

```python
for epoch in range(5):
    print(epoch)
    for i in range(x_train.shape[0]):
        nn.train(x_train[i], y_train[i])
```

ここではepochという変数を使うループが増えています。このように繰り返し訓練を行う仕組みはエポックと呼ばれます。ニューラルネットワークではエポック数を増やし、繰り返し学習を行うことで精度の向上を期待できます。一方でエポック数を増やしすぎると学習に時間がかかったり、過学習の原因となったりすることもあります。
