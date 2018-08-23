from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from ray.rllib.models.model import Model
from ray.rllib.models.misc import get_activation_fn, flatten, AddCoords
import numpy as np

class VisionNetwork(Model):
    """Generic vision network."""

    def _build_layers(self, inputs, num_outputs, options):
        #print("visionnet: testing inputs size {}".format(np.shape(inputs)))
        if options.get('custom_options', {}).get('add_coordinates'):
            with_r = False
            if options.get('custom_options', {}).get('add_coords_with_r'):
                with_r = True
            #addcoords = AddCoords(x_dim=x_dim, y_dim=y_dim, with_r=with_r)
            #addcoords = AddCoords(x_dim=tf.shape(inputs)[0],y_dim=tf.shape(inputs)[1],with_r=with_r)
            addcoords = AddCoords(x_dim=int(np.shape(inputs)[1]), y_dim=int(np.shape(inputs)[1]),with_r=with_r)
            inputs = addcoords(inputs)
            print("visionnet: Added coordinate filters tensor size is now {}".format(np.shape(inputs)))

        filters = options.get("conv_filters")
        if not filters:
            filters = get_filter_config(options)

        activation = get_activation_fn(options.get("conv_activation", "relu"))

        with tf.name_scope("vision_net"):
            for i, (out_size, kernel, stride) in enumerate(filters[:-1], 1):
                inputs = slim.conv2d(
                    inputs,
                    out_size,
                    kernel,
                    stride,
                    activation_fn=activation,
                    scope="conv{}".format(i))
            out_size, kernel, stride = filters[-1]
            fc1 = slim.conv2d(
                inputs,
                out_size,
                kernel,
                stride,
                activation_fn=activation,
                padding="VALID",
                scope="fc1")
            if tf.__version__ == '1.4.0':
                fc2 = slim.conv2d(
                    fc1,
                    num_outputs,
                    1,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope="fc2")
            else:
                fc2 = slim.conv2d(
                    fc1,
                    num_outputs,
                    [1, 1],
                    activation_fn=None,
                    normalizer_fn=None,
                    scope="fc2")
            return flatten(fc2), flatten(fc1)


def get_filter_config(options):
    if tf.__version__ == '1.4.0':
        filters_80x80 = [
            [16, 8, 4],
            [32, 4, 2],
            [512, 10, 1],
        ]
        filters_42x42 = [
            [16, 4, 2],
            [32, 4, 2],
            [512, 11, 1],
        ]
    else:
        filters_80x80 = [
            [16, [8, 8], 4],
            [32, [4, 4], 2],
            [512, [10, 10], 1],
        ]
        filters_42x42 = [
            [16, [4, 4], 2],
            [32, [4, 4], 2],
            [512, [11, 11], 1],
        ]
    dim = options.get("dim", 80)
    if dim == 80:
        return filters_80x80
    elif dim == 42:
        return filters_42x42
    else:
        raise ValueError(
            "No default configuration for image size={}".format(dim) +
            ", you must specify `conv_filters` manually as a model option.")
