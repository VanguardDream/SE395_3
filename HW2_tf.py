import tensorflow as tf
import functions as func
import numpy as np
import matplotlib.pyplot as plt

lossgraph = True

# Loader
data_train, data_test = tf.keras.datasets.mnist.load_data()

(image_train, label_train) = data_train
(image_test, label_test) = data_test

image_train = image_train.reshape(image_train.shape[0], 28, 28, 1)
image_test = image_test.reshape(image_test.shape[0], 28, 28, 1)

image_train = image_train.astype('float32')
image_test = image_test.astype('float32')

# # Regularize
# (image_train, label_train) = (image_train/255.0, label_train/255.0)
# (image_test, label_test) = (image_test/255.0, label_test/255.0)

# Model structure - 2 Layer CNN
two_layer_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

two_layer_model.compile(
    optimizer='SGD',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

hist = two_layer_model.fit(image_train, 
    label_train, 
    batch_size=16, 
    epochs=10, 
    validation_data=(image_test, label_test)
)

if lossgraph:
    fig, loss_ax = plt.subplots()

    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'b', label='validation loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')

    
    acc_ax.set_ylabel('accuracy')
    acc_ax.plot(hist.history['val_accuracy'], 'r', label ='Validation Accuracy')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    plt.show()
