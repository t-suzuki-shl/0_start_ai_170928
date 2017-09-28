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
