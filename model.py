"""
model.py
"""

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import convolution

def flood_filling_network(inp, num_modules, ksize):
    """Flood filling network model.

    Args:
        inp: 5D Tensor (batch_size, depth, width, height, channels)
            the input tensor
        num_modules: int
            the number of convolutional modules to use (> 2)
         ksize: (x, y, z) int
            the dimensions of the kernel
    Returns:
        out: 5D Tensor (batch_size, depth, width, height, channels)
            the output tensor
    """
    assert num_modules > 2, "num_modules should be greater than 2"
    #Compute first convolution with inp channel = 1
    conv = convolution(inp, 32, ksize, activation_fn = tf.nn.relu)
    for _ in range(num_modules - 1):
        conv1 = convolution(conv, 32, ksize, activation_fn = tf.nn.relu)
        conv2 = convolution(conv1, 32, ksize, activation_fn = None)
        conv = tf.nn.relu(conv + conv2)
    conv = convolution(conv, 32, ksize, activation_fn = tf.nn.relu)
    out = convolution(conv, 1, ksize, activation_fn = None)
    return out



