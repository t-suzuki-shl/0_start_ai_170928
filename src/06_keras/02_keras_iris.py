from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)

# one-hot label
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

model = Sequential()
model.add(Dense(50, input_dim=4))
model.add(Activation('sigmoid'))
model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=20, batch_size=1)

score = model.evaluate(x_test, y_test, verbose=0)
print("test acc : ", score[1])

# import matplotlib.pyplot as plt
#
# plt.ylim()
# plt.plot(history.history['acc'], label="acc")
# plt.plot(history.history['loss'],label="loss")
# plt.legend()
# plt.show()
