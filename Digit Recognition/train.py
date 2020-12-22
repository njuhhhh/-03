# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np


# download the data
mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

train_images = train_images / 255.0
test_images = test_images / 255.0


def create_model():
    # It's necessary to give the input_shape，or it will fail when you load the model
    # The error will be like : You are trying to load the 4 layer models to the 0 layer
    model = keras.Sequential([
        keras.layers.Conv2D(32, [5, 5], activation=tf.nn.relu, input_shape=(28, 28, 1)),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(64, [7, 7], activation=tf.nn.relu),
        keras.layers.MaxPool2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(576, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# reshape the shape before using it, for that the input of cnn is 4 dimensions
train_images = np.reshape(train_images, [-1, 28, 28, 1])
test_images = np.reshape(test_images, [-1, 28, 28, 1])

# train
model = create_model()
model.fit(train_images, train_labels, epochs=4)

# save the model
model.save('my_model.h5')

# Evaluate
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print('Test accuracy:', test_acc)