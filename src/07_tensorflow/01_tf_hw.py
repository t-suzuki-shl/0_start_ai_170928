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
