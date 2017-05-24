"""
model.py
"""

import tensorflow as tf

def flood_filling_network(inp, num_modules):
    """Flood filling network model.

    Args:
        inp: 5D Tensor (batch_size, depth, width, height, channels)
            the input tensor
        num_modules: int
            the number of convolutional modules to use (> 2)
    Returns:
        out: 5D Tensor (batch_size, depth, width, height, channels)
            the output tensor
    """
    assert(num_modules > 2, "num_modules should be greater than 2")
    #Compute first convolution with inp channel = 1
    inp = conv_module(inp, 1, 32)
    for _ in range(num_modules - 2):
        inp = conv_module(inp, 32, 32)
    out = conv_module(inp, 32, 1)
    return out

def conv_module(inp, channel_in, channel_out):
    """Implements a convolutional module as outlined in the paper.
    That is: two 3D convolutions, with an relu activation on the first
    one and an relu activation over the sum of the orginal input and
    the output of the second convolution.

    Args:
        inp: 5D Tensor (batch_size, depth, width, height, channels)
            the input data
        channel_in: int
            the number of input channels
        channel_out: int
            the number of output channels
    Returns:
        conv: 5D Tensor (batch_size, depth, width, height, channels)
            the output tensor
    """
    conv1 = tf.nn.relu(conv_3d(inp, channel_in, channel_out))
    conv2 = conv_3d(conv1, channel_out, channel_out)
    return tf.nn.relu(inp + conv2)

def conv_3d(inp, channel_in, channel_out):
    """Implements a 3d convolution with no specified activation.

    Args:
        x: 5D Tensor (batch_size, depth, width, height, channels)
            the input data
        channel_in: int
            the number of input channels
        channel_out: int
            the number of output channels
    Returns:
        conv: 5D Tensor (batch_size, depth, width, height, channels)
            the output tensor
    """
    weight_shape = [3, 3, 3, channel_in, channel_out]
    W = tf.Variable(tf.random_normal(weight_shape))
    b = tf.Variable(tf.random_normal(channel_out))
    return tf.nn.conv3d(inp, W, [1, 1, 1, 1, 1], 'SAME') + b
