import tensorflow as tf
import functions as func
import numpy as np
import matplotlib.pyplot as plt
import datetime

lossgraph = True
numoflayer = 2

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


if numoflayer == 2:
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

    # Tensorboard log directory
    log_dir = "log/two_l/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    hist = two_layer_model.fit(
        image_train, 
        label_train, 
        batch_size=16, 
        epochs=10, 
        validation_data=(image_test, label_test),
        callbacks=[tensorboard_callback]
    )

    # # Print Confusion Matrix
    # conf_mat = func.confusion(np.argmax(two_layer_model.predict(image_test), axis=-1),data_test)
    # print(conf_mat)


else:
    # Model structure - 3 Layer CNN
    three_layer_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(32,activation='relu'),
        tf.keras.layers.Dense(10,activation='softmax')
    ])

    three_layer_model.compile(
    optimizer='SGD',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )
    
    # Tensorboard log directory
    log_dir = "log/three_l/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    hist = three_layer_model.fit(
        image_train,
        label_train,
        batch_size=16,
        epochs=10,
        validation_data=(image_test, label_test),
        callbacks=[tensorboard_callback]
    )


# Plot Loss Graph
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

# Tensorboard image upload
img = np.reshape(image_train[123], (-1, 28, 28, 1))

file_writer = tf.summary.create_file_writer(log_dir)

with file_writer.as_default():
    tf.summary.image("Training data", img, step=0)