"""Networks architectures for CARLA Agent"""

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


# TODO: remove recurrences, make "auto-regressive"
def feature_net(input_layer: Input, noise=0.05, num_layers=2, units=32, activation='swish', dropout=0.5,
                name='feature_out') -> Layer:
    """Network for vector features"""
    x = GaussianNoise(stddev=noise)(input_layer)
    x = tf.expand_dims(x, axis=0)

    # recurrence
    x, state = GRU(units, stateful=True, return_state=True)(x)
    x = subtract([x, state])
    x = LayerNormalization()(x)

    # dense layers
    for _ in range(num_layers):
        x = Dense(units, activation=activation)(x)
        x = Dropout(rate=dropout)(x)

    return add([x, state], name=name)


def convolutional(input_layer: Input, filters=32, filters_multiplier=1.0, blocks=2, units=64, dropout=0.2,
                  kernel=(3, 3), strides=(2, 2), activation='swish', padding='same', name='conv_out'):
    """Convolutional neural network"""
    def block(prev_layer, i):
        h = DepthwiseConv2D(kernel_size=kernel, padding=padding, activation=activation)(prev_layer)
        h = Conv2D(filters=int(filters * i * filters_multiplier), kernel_size=kernel, padding=padding)(h)
        h = ReLU(max_value=6.0, negative_slope=0.2)(h)
        h = SpatialDropout2D(rate=dropout)(h)
        h = MaxPooling2D(pool_size=3, strides=strides)(h)  # overlapping max-pool
        return h

    x = input_layer
    k = 0
    for _ in range(blocks):
        x1 = block(x, i=k + 1)
        x2 = DepthwiseConv2D(kernel_size=kernel, padding='same', activation='swish')(x1)
        k += 2
        x = concatenate([x1, x2], axis=-1)  # depth-concat
        x = LayerNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    x = LayerNormalization()(x)
    x = tf.expand_dims(x, axis=0)

    # 1-timestep RNN
    x, state = GRU(units, stateful=True, return_state=True)(x)
    return subtract([x, state], name=name)
