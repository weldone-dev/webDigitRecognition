import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image

import numpy as np
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = np.around(x_train / 255.0+0.4)
x_test = np.around(x_test / 255.0+0.4)

def initModel():
    model = Sequential([
      Flatten(input_shape=(28, 28)),
      Dense(128, activation='relu'),
      Dropout(0.5),
      Dense(10, activation='softmax')
    ])
    return model
model = initModel()
#model = load_model('my_model.h5')

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.evaluate(x_test, y_test))
model.fit(x_train, y_train, epochs=5)
model.save('my_model.h5')
