import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation

x_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
y_train = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

x_test = np.array([[0.1, 0.3, 0.5]])
y_test = np.array([[0, 0, 1]])

model = Sequential()
model.add(Dense(2, input_dim=3))
model.add(Activation('sigmoid'))
model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train)

print(model.evaluate(x_test, y_test, verbose=0))
