## 3.5 ニューラルネットワークの実装 - PART2

引き続きニューラルネットワークを構成する要素を見ていきましょう。これまでに活性化関数を使って入力信号を出力信号に変換する流れを見てきました。次に教師データを活用して、重みやバイアスを改善していく手順を見ていきましょう。

+ 損失関数
  + 2乗和誤差
  + クロスエントロピー誤差

## 3.6 損失関数

損失関数はニューラルネットワークの出力と、教師データの相違を算出する関数です。ニューラルネットワークに訓練データを投入して、損失関数の結果を改善していくことで、ニューラルネットワークの精度を高めていきます。

<img src="img/03_09.png?abcs" width="600px">

> 損失関数は図の緑色の部分で動作します。

<div style="page-break-before:always"></div>


### 2乗和誤差

まずはシンプルな損失関数である2乗和誤差を実装してみましょう。2乗和誤差は教師データ（t）とニューラルネットワークの出力データ（y）の差分をとります。それから差分を2乗し、総和を求めます。次のように実装できます。

```python
def mean_squared(y, t):
    return 0.5 * np.sum((y - t)**2)
```

> 0.5を乗算しているのはあとの勾配（微分）の算出を効率よく行うためです。

実際に動作確認してみましょう。

```python
import numpy as np

def mean_squared(y, t):
    return 0.5 * np.sum((y - t)**2)

y = np.array([0.1, 0.1, 0.8])
t = np.array([0, 0, 1])
print(mean_squared(y, t))

y = np.array([0.1, 0.1, 0.8])
t = np.array([0, 1, 0])
print(mean_squared(y, t))
```

プログラムの出力結果は次のようになるでしょう。

```
0.03
0.73
```

値の大小に注目してください。教師データと予測値（出力結果の最大値）が等しい場合（1つ目の出力）は値が 0.03 と小さい値になっています。一方、教師データと予測値が等しくない場合（2つ目の出力）は0.73と大きな値になっています。ニューラルネットワークは学習を重ねることで、この損失関数の値を小さくしていくことが目的となります。

<div style="page-break-before:always"></div>


### クロスエントロピー誤差

損失関数にはクロスエントロピー誤差を利用することもできます。クロスエントロピー誤差では教師データに期待値の対数を掛け合わせたものの総和をとります。


```python
def cross_entropy(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
```

> 変数deltaは微小なデータを表します。対数関数（log）は引数に0を受け取ってしまうと無限となってしまうためです。

実際に動作確認してみましょう。

```python
import numpy as np

def cross_entropy(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

y = np.array([0.1, 0.1, 0.8])
t = np.array([0, 0, 1])
print(cross_entropy(y, t))

y = np.array([0.1, 0.1, 0.8])
t = np.array([0, 1, 0])
print(cross_entropy(y, t))
```

プログラムの出力結果は次のようになるでしょう。

```
0.223143426314
2.30258409299
```

2乗和誤差の場合と同じく、教師データと予測値が等しい場合は、損失関数の結果は小さくなり、等しくない場合は損失関数の結果は大きくなっているのがわかります。

> 以降はクロスエントロピー誤差を損失関数に使います。

<div style="page-break-before:always"></div>


### これまでのまとめ

ニューラルネットワークの出力結果と教師データを比較するために損失関数を学びました。損失関数には2乗和誤差やクロスエントロピー誤差といったアルゴリズムがありました。これらの損失関数の出力結果を最小にすることが、ニューラルネットワークを学習するポイントです。

それでは実際にニューラルネットワークのプログラムを続きを作成してみましょう。既存のNeuralNetworkクラスを以下のとおり修正します。

+ 損失関数 cross_entropyメソッドを追加
+ 既存のinputメソッドの修正
  + 引数に教師データ t を追加
  + 損失関数の呼び出しを実装
  + 戻り値で損失関数の結果を返却


<div style="page-break-before:always"></div>

```python
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

nn = NeuralNetwork()
x = np.array([0.5, 0.6, 0.7])
y = np.array([0, 0, 1])

print(nn.input(x, y))
```

プログラムの実行結果は次のようになります。

```
0.770442010864
```

ここでは損失関数の出力結果を求めました。以降は、損失関数の出力結果を最小化する重みやバイアスを決定する方法を学びます。
