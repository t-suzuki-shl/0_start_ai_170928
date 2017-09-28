## 4.8 （参考）TensorFlowチュートリアル - 1 TensorFlow入門

TensorFlowを使った機械学習のサンプルプログラムも見てみましょう。

次のプログラムは線形回帰を行うプログラムです。y = W \* x + b という数式モデルに対して、損失関数を最小とする W、b を最適化処理（オプティマイザー）によって求めることができます。

> TensorFlowはディープラーニングに特化したフレームワークではありません。このようなシンプルな線形回帰式を解くこともできます。

<div style="page-break-before:always"></div>

```python
import tensorflow as tf

# 学習するパラメータ
W = tf.Variable([0.5], dtype=tf.float32)
b = tf.Variable([0.5], dtype=tf.float32)

# 入力データ（x）と出力データ（y）
# 入力データ（x）、数式モデル（linear_model）、期待値データ（y）
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# 損失関数（2乗和誤差）
loss = tf.reduce_sum(tf.square(linear_model - y))

# オプティマイザー
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# 訓練データ
x_train = [0, 1, 2, 3, 4]
y_train = [5, 7, 9, 11, 13]

# 学習（訓練）
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
  sess.run(train, {x: x_train, y: y_train})
  # print(sess.run([W]))

# 結果の表示
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
```

<div style="page-break-before:always"></div>

プログラムの実行結果は次のようになります。

```python
2017-08-25 13:35:12.558036: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorF
low library wasn't compiled to use SSE4.1 instructions, but these are available on your mac
hine and could speed up CPU computations.
2017-08-25 13:35:12.558104: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorF
low library wasn't compiled to use SSE4.2 instructions, but these are available on your mac
hine and could speed up CPU computations.
2017-08-25 13:35:12.558126: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorF
low library wasn't compiled to use AVX instructions, but these are available on your machin
e and could speed up CPU computations.
2017-08-25 13:35:12.558132: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorF
low library wasn't compiled to use AVX2 instructions, but these are available on your machi
ne and could speed up CPU computations.
2017-08-25 13:35:12.558137: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorF
low library wasn't compiled to use FMA instructions, but these are available on your machin
e and could speed up CPU computations.
W: [ 2.00000262] b: [ 4.99999237] loss: 9.27685e-11
```

出力の最後の部分で、W、B が出力されているのがわかります。

> y = 2 \* x + 5 という数式は訓練データと見合っているのがわかります。

<div style="page-break-before:always"></div>

プログラムの詳細を見てみましょう。

```python
# 学習するパラメータ
W = tf.Variable([0.5], dtype=tf.float32)
b = tf.Variable([0.5], dtype=tf.float32)
```

TensorFlowは学習するパラメータをtf.Variable()メソッドで定義します。引数には初期値とデータ型を定義しています。

> 変数 W や b にはtf.Variableオブジェクトが代入されます。

次にTensorFlowでは求めたい数式モデルと入力データ、期待値データを定義します。

```python
# 入力データ（x）、数式モデル（linear_model）、期待値データ（y）
# y = W * x + b
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
```

数式モデルの入力データや期待値データはtf.placeholderメソッドで定義します。定義した数式モデル（linear_model）は、後のtf.Sessionオブジェクトを通じて実行します。

> 変数 x や y には tf.Tensor オブジェクトが代入されます。

次に損失関数を定義します。

```python
# 損失関数（2乗和誤差）
loss = tf.reduce_sum(tf.square(linear_model - y))
```

> 変数 loss には tf.Tensor オブジェクトが代入されます。

損失関数は、モデルの出力データと期待値データから損失を算出します。ここでは2乗和誤差を実装しています。

> 損失関数には他にもクロスエントロピー誤差などがあります。

<div style="page-break-before:always"></div>


次に最適化処理（オプティマイザー）を定義しています。

```python
# オプティマイザー（勾配降下法）
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
```

tf.train.GradientDescentOptimizerメソッドによって、勾配降下法を表現するオプティマイザーが生成されます。引数に指定した0.01は学習率を意味します。

> ここで作成した train 変数を使って学習（訓練）を行います。

次に訓練データを定義しています。

```python
# 訓練データ
x_train = [0, 1, 2, 3, 4]
y_train = [5, 7, 9, 11, 13]
```

以上で学習に必要な準備が整いました。


```python
# 学習（訓練）
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
  sess.run(train, {x: x_train, y: y_train})
  # print(sess.run([W]))
```

TensorFlowでは、定義したパラメータ（Variable）を初期化するためにtf.global_variables_initializerメソッドを呼び出す必要があります。厳密にはこの時点ではパラメータの初期化は実行されず、後で作成するTensorFlowのセッションを通じてパラメータの初期化を実行します。

ここからは実際に訓練データを投入して学習をスタートします。まずはtf.Sessionメソッドを呼び出してセッションを開始します。TensorFlowではセッション（sess変数）のrunメソッドを通じて、パラメータの初期化や、学習、評価を行います。ここではセッションを通じて変数を初期化した後、1000回の学習ループを実行します。実際に学習を行うのは sess.run(train, {x: x_train, y: y_train}) の部分です。引数に指定した train を実行するには、入力データ x と期待値データ y 2つのプレースホルダに訓練データを指定します。

学習を終えたらパラメータ w、b 損失関数の値をそれぞれ出力してみましょう。

```python
# 結果の表示
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
```

勾配降下法によって算出された値を確認することができるでしょう。


<div style="page-break-before:always"></div>

## 4.9 （参考）TensorFlowチュートリアル - 2 MNISTデータの検証

次にMNISTデータを処理するTensorFlowのサンプルプログラムを見てみましょう。

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 入力層（784）、出力層（10）
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 実測値
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 期待値
y_ = tf.placeholder(tf.float32, [None, 10])

# 損失関数（クロスエントロピー誤差）
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=1))

# オプティマイザー（勾配降下法）
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 学習（訓練）
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(10):
    for _ in range(1000):
      batch_xs, batch_ys = mnist.train.next_batch(32)
      sess.run(train_step, {x: batch_xs, y_: batch_ys})
    # print(sess.run([b]))

# 正答率（accuracy）の表示
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, {x: mnist.test.images, y_: mnist.test.labels}))
```

<div style="page-break-before:always"></div>

プログラムの実行結果は次のようになります。

```python
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
2017-08-25 14:37:36.609146: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorF
low library wasn't compiled to use SSE4.1 instructions, but these are available on your mac
hine and could speed up CPU computations.
2017-08-25 14:37:36.609170: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorF
low library wasn't compiled to use SSE4.2 instructions, but these are available on your mac
hine and could speed up CPU computations.
2017-08-25 14:37:36.609176: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorF
low library wasn't compiled to use AVX instructions, but these are available on your machin
e and could speed up CPU computations.
2017-08-25 14:37:36.609180: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorF
low library wasn't compiled to use AVX2 instructions, but these are available on your machi
ne and could speed up CPU computations.
2017-08-25 14:37:36.609186: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorF
low library wasn't compiled to use FMA instructions, but these are available on your machin
e and could speed up CPU computations.
0.9122
```

出力の最終行に正答率 91% が出力されているのがわかります。

> 初回実行時にはMNISTデータのダウンロードが発生します。

<div style="page-break-before:always"></div>

プログラムの詳細を見てみましょう。

冒頭の部分でMNISTデータを取得しています。

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

上記の例ではカレントフォルダにMNIST_dataフォルダを作成して、MNISTデータを保存しています。また教師データの期待値はone-hotラベル表現として保存しています。

次に検証する数式モデルを定義しています。

```python
# 入力層（784）、出力層（10）
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 実測値
y = tf.nn.softmax(tf.matmul(x, W) + b)
```

ここでは入力層と出力層からなるニューラルネットワークを定義しています。また出力となる実測値はsoftmax関数で確率に変換しています。なおtf.matmulメソッドはnumpy.dotメソッドと同様に行列の積を算出します。

次に訓練データの期待値を入力するプレースホルダを定義しています。

```python
# 期待値
y_ = tf.placeholder(tf.float32, [None, 10])
```

上記の期待値 y_ とモデルの実測値 y の誤差を求めるものが次の損失関数です。

```python
# 損失関数（クロスエントロピー誤差）
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=1))
```

ここではクロスエントロピー誤差を算出しています。

<div style="page-break-before:always"></div>


次にオプティマイザーの定義です。

```python
# オプティマイザー（勾配降下法）
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```

学習率に0.5を定義し勾配降下法のオプティマイザーを定義しています。

> TensorFlowで利用可能なオプティマイザーは以下のURLにまとめられています。 https://www.tensorflow.org/api_guides/python/train

次にTensorFlowのセッションを開始して、訓練データを投入して学習をスタートします。

```python
# 学習（訓練）
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(10):
    for _ in range(1000):
      batch_xs, batch_ys = mnist.train.next_batch(32)
      sess.run(train_step, {x: batch_xs, y_: batch_ys})
    # print(sess.run([b]))
```

ここではMNISTの訓練データの中から32件ずつデータを取り出し、訓練を実施しています。

最後に訓練の結果を確認しています。

```python
# 正答率（accuracy）の表示
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, {x: mnist.test.images, y_: mnist.test.labels}))
```

ここでは正答率を出力するためにグラフを定義しています。具体的には実測値 y と期待値 y_ の等しいものを算出するcorrect_predictionノードを定義し、tf.reduce_meanメソッドによってcorrect_predictionの結果の平均値を算出しています。
