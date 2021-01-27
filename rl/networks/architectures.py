"""Pre-defined network architectures ready to be used as part of Agent's networks"""

import tensorflow as tf

from tensorflow.keras.layers import *


def shufflenet_v2(input_image: Input, g=1.0, leak=0.0, last_channels=1024, linear_units=128):
    """ShuffleNet-V2, based on:
       https://github.com/tensorpack/tensorpack/blob/master/examples/ImageNetModels/shufflenet.py
    """
    assert g in [0.5, 1.0, 1.5, 2.0]

    channels = {0.5: [48, 96, 192],
                1.0: [116, 232, 464],
                1.5: [176, 352, 704],
                2.0: [244, 488, 976],
                #
                0.75: [82, 164, 328],
                1.25: [146, 292, 584],
                1.75: [210, 420, 840]}

    def activation(layer: Layer):
        """Activation: BatchNormalization + ReLU6"""
        layer = BatchNormalization()(layer)
        layer = ReLU(max_value=6.0, negative_slope=leak)(layer)
        return layer

    def channel_shuffle(layer: Layer, groups=2):
        in_shape = layer.get_shape().as_list()
        in_channel = in_shape[-1]
        assert in_channel % groups == 0, in_channel

        # (batch, h, w, c, group)
        layer = tf.reshape(layer, [-1, in_shape[1], in_shape[2], in_channel // groups, groups])
        layer = tf.transpose(layer, [0, 1, 2, 4, 3])
        layer = tf.reshape(layer, [-1, in_shape[1], in_shape[2], in_channel])
        return layer

    def shufflenet_v2_unit(layer: Layer, num_channels: int, stride: int):
        # channel split:
        if stride == 1:
            shortcut, layer = tf.split(layer, 2, axis=-1)
        else:
            shortcut, layer = layer, layer

        shortcut_channels = int(shortcut.shape[-1])

        # 1x1 pointwise conv -> 3x3 depthwise conv -> batch-norm -> 1x1 conv
        layer = Conv2D(num_channels // 2, kernel_size=1, padding='same')(layer)
        layer = activation(layer)
        layer = DepthwiseConv2D(kernel_size=3, strides=stride, padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2D(num_channels - shortcut_channels, kernel_size=1, padding='same')(layer)
        layer = activation(layer)

        if stride == 2:
            shortcut = DepthwiseConv2D(kernel_size=3, strides=2, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)
            shortcut = Conv2D(shortcut_channels, kernel_size=1, padding='same')(shortcut)
            shortcut = activation(shortcut)

        output = tf.concat([shortcut, layer], axis=-1)
        output = channel_shuffle(output)
        return output

    def shufflenet_stage(layer: Layer, num_channels: int, num_blocks: int):
        for i in range(num_blocks):
            layer = shufflenet_v2_unit(layer, num_channels, stride=2 if i == 0 else 1)
        return layer

    # Input
    x = Conv2D(24, kernel_size=3, strides=2)(input_image)
    x = activation(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Stages
    c1, c2, c3 = channels[g]
    x = shufflenet_stage(x, num_channels=c1, num_blocks=4)
    x = shufflenet_stage(x, num_channels=c2, num_blocks=8)
    x = shufflenet_stage(x, num_channels=c3, num_blocks=4)

    # Output
    x = Conv2D(last_channels, kernel_size=1)(x)
    x = activation(x)
    x = GlobalAveragePooling2D()(x)

    if linear_units > 0:
        return Dense(units=linear_units, activation='linear')(x)

    return x
