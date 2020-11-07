import numpy as np

def padding(X, pad):
    output = np.pad(X, ((0,0),(pad,pad),(pad,pad),(0,0)), 'constant', constant_values=0)

    return output

def conv_