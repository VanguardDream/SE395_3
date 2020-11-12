import numpy as np
import tensorflow as tf

def padding(X, pad):
    output = np.pad(X, ((0,0),(pad,pad),(pad,pad),(0,0)), 'constant', constant_values=0)

    return output

def conv_single_step(a, W, b):
    s = np.multiply(a,W) + b
    output = np.sum(s)

    return output

# def conv_forward(a, W, b, params):
#     (m, hight, width, color) = a.shape

def confusion(y_pred, test_set):
    (test_image, test_label) = test_set
    confusion_mat = tf.math.confusion_matrix(labels = test_label, predictions = y_pred)
    return confusion_mat