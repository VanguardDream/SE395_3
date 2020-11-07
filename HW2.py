import tensorflow as tf
import functions as func
import numpy as np

# Loader
data_train, data_test = tf.keras.datasets.mnist.load_data()

(image_train, label_train) = data_train
(image_test, label_test) = data_test

x = np.ones((4,3,3,2))
x_pad = func.padding(x,2)

print(x_pad.shape)
print(x_pad)