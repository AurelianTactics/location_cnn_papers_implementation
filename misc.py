from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.layers import base #added

def normc_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


def get_activation_fn(name):
    return getattr(tf.nn, name)


def conv2d(x,
           num_filters,
           name,
           filter_size=(3, 3),
           stride=(1, 1),
           pad="SAME",
           dtype=tf.float32,
           collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [
            filter_size[0], filter_size[1],
            int(x.get_shape()[3]), num_filters
        ]

        # There are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit.
        fan_in = np.prod(filter_shape[:3])
        # Each unit in the lower layer receives a gradient from: "num output
        # feature maps * filter height * filter width" / pooling size.
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # Initialize weights with random weights.
        w_bound = np.sqrt(6 / (fan_in + fan_out))

        w = tf.get_variable(
            "W",
            filter_shape,
            dtype,
            tf.random_uniform_initializer(-w_bound, w_bound),
            collections=collections)
        b = tf.get_variable(
            "b", [1, 1, 1, num_filters],
            initializer=tf.constant_initializer(0.0),
            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b


def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(
        name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(
        name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b


def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

#added
class AddCoords(base.Layer):
    """Add coords to a tensor"""
    def __init__(self, x_dim=64, y_dim=64, with_r=False):
        super(AddCoords, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r

    def call(self, input_tensor):
        """
        input_tensor: (batch, x_dim, y_dim, c)
        """
        batch_size_tensor = tf.shape(input_tensor)[0]
        xx_ones = tf.ones([batch_size_tensor, self.x_dim],dtype=tf.int32)
        xx_ones = tf.expand_dims(xx_ones, -1)
        xx_range = tf.tile(tf.expand_dims(tf.range(self.x_dim), 0),[batch_size_tensor, 1])
        xx_range = tf.expand_dims(xx_range, 1)
        xx_channel = tf.matmul(xx_ones, xx_range)
        xx_channel = tf.expand_dims(xx_channel, -1)

        yy_ones = tf.ones([batch_size_tensor, self.y_dim],dtype=tf.int32)
        yy_ones = tf.expand_dims(yy_ones, 1)
        yy_range = tf.tile(tf.expand_dims(tf.range(self.y_dim), 0),[batch_size_tensor, 1])
        yy_range = tf.expand_dims(yy_range, -1)
        yy_channel = tf.matmul(yy_range, yy_ones)
        yy_channel = tf.expand_dims(yy_channel, -1)
        xx_channel = tf.cast(xx_channel, 'float32') / (self.x_dim - 1)
        yy_channel = tf.cast(yy_channel, 'float32') / (self.y_dim - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        ret = tf.concat([input_tensor, xx_channel,yy_channel], axis=-1)

        if self.with_r:
            rr = tf.sqrt(tf.square(xx_channel - 0.5) + tf.square(yy_channel - 0.5))
            ret = tf.concat([ret, rr], axis=-1)

        return ret


class CoordConv(base.Layer):
    """CoordConv layer as in the paper."""
    def __init__(self, x_dim, y_dim, with_r, *args, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(x_dim=x_dim,y_dim=y_dim,with_r=with_r)
        self.conv = tf.layers.Conv2D(*args, **kwargs)

    def call(self, input_tensor):
        ret = self.addcoords(input_tensor)
        ret = self.conv(ret)
        return ret