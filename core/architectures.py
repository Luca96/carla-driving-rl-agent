"""Pre-defined Network's architectures that operate over 'time'"""

import tensorflow as tf

from tensorflow.keras.layers import *
from typing import List


def feature_net(inputs: Input, time_horizon: int, units=32, num_layers=2, activation='relu',
                normalization=None) -> List[Layer]:
    splits = tf.split(inputs, time_horizon, axis=1)
    splits = [tf.squeeze(split, axis=1) for split in splits]

    if normalization == 'batch':
        batch_norm = BatchNormalization()
        x = [batch_norm(_x) for _x in splits]
    else:
        x = splits

    dense = [Dense(units, activation=activation, bias_initializer='glorot_uniform') for _ in range(num_layers)]
    batch_norm = [BatchNormalization() for _ in range(num_layers)]

    for i in range(num_layers):
        x = [dense[i](_x) for _x in x]
        x = [batch_norm[i](_x) for _x in x]

    return x


def shufflenet_v2(inputs: Input, time_horizon: int, g=1.0, leak=0.0, last_channels=1024) -> List[Layer]:
    assert g in [0.5, 1.0, 1.5, 2.0]

    channels = {0.5: [48, 96, 192],
                1.0: [116, 232, 464],
                1.5: [176, 352, 704],
                2.0: [244, 488, 976],
                #
                0.75: [82, 164, 328],
                1.25: [146, 292, 584],
                1.75: [210, 420, 840]}

    # --- (Re-)define NN's shared (same weights) layers ---

    def activation_fn():
        def create_once():
            batch_normalization = BatchNormalization()
            relu = ReLU(max_value=6.0, negative_slope=leak)
            return lambda layers: [relu(batch_normalization(layer)) for layer in layers]

        return create_once()

    def batch_norm():
        def create_once():
            batch_normalization = BatchNormalization()
            return lambda layers: [batch_normalization(layer) for layer in layers]

        return create_once()

    def conv2d(**kwargs):
        def create_once():
            conv = Conv2D(**kwargs)
            return lambda layers: [conv(layer) for layer in layers]

        return create_once()

    def depthwise_conv2d(**kwargs):
        def create_once():
            depth_conv = DepthwiseConv2D(**kwargs)
            return lambda layers: [depth_conv(layer) for layer in layers]

        return create_once()

    def max_pool2d(**kwargs):
        def create_once():
            max_pool = MaxPooling2D(**kwargs)
            return lambda layers: [max_pool(layer) for layer in layers]

        return create_once()

    def global_avg_pool2d():
        def create_once():
            global_pool2d = GlobalAveragePooling2D()
            return lambda layers: [global_pool2d(layer) for layer in layers]

        return create_once()

    def tf_split(layers: List[Layer]) -> (List[Layer], List[Layer]):
        shortcuts = []
        _layers = []

        for layer in layers:
            shortcut, _layer = tf.split(layer, 2, axis=-1)
            shortcuts.append(shortcut)
            _layers.append(_layer)

        return shortcuts, _layers

    def tf_concat(shortcuts: List[Layer], layers: List[Layer]) -> List[Layer]:
        results = []

        for shortcut, layer in zip(shortcuts, layers):
            result = tf.concat([shortcut, layer], axis=-1)
            results.append(result)

        return results

    # --- ShuffleNet's building blocks ---

    def channel_shuffle(layer: Layer, groups=2):
        in_shape = layer.get_shape().as_list()
        in_channel = in_shape[-1]
        assert in_channel % groups == 0, in_channel

        # (batch, h, w, c, group)
        layer = tf.reshape(layer, [-1, in_shape[1], in_shape[2], in_channel // groups, groups])
        layer = tf.transpose(layer, [0, 1, 2, 4, 3])
        layer = tf.reshape(layer, [-1, in_shape[1], in_shape[2], in_channel])
        return layer

    def shufflenet_v2_unit(layers: List[Layer], num_channels: int, stride: int) -> List[Layer]:
        # channel split:
        if stride == 1:
            shortcuts, layers = tf_split(layers)
        else:
            shortcuts, layers = layers, layers

        shortcut_channels = int(shortcuts[0].shape[-1])

        # 1x1 pointwise conv -> 3x3 depthwise conv -> batch-norm -> 1x1 conv
        layers = conv2d(filters=num_channels // 2, kernel_size=1, padding='same')(layers)
        layers = activation_fn()(layers)
        layers = depthwise_conv2d(kernel_size=3, strides=stride, padding='same')(layers)
        layers = batch_norm()(layers)
        layers = conv2d(filters=num_channels - shortcut_channels, kernel_size=1, padding='same')(layers)
        layers = activation_fn()(layers)

        if stride == 2:
            shortcuts = depthwise_conv2d(kernel_size=3, strides=2, padding='same')(shortcuts)
            shortcuts = batch_norm()(shortcuts)
            shortcuts = conv2d(filters=shortcut_channels, kernel_size=1, padding='same')(shortcuts)
            shortcuts = activation_fn()(shortcuts)

        outputs = tf_concat(shortcuts, layers)
        outputs = [channel_shuffle(layer) for layer in outputs]
        return outputs

    def shufflenet_stage(layers: List[Layer], num_channels: int, num_blocks: int):
        for i in range(num_blocks):
            layers = shufflenet_v2_unit(layers, num_channels, stride=2 if i == 0 else 1)

        return layers

    # --- Build the network ---
    # Prepare inputs: split input tensor over time, and remove (squeeze) the time axis
    splits = tf.split(inputs, time_horizon, axis=1)
    splits = [tf.squeeze(split, axis=1) for split in splits]

    # Input
    x = conv2d(filters=24, kernel_size=3, strides=2)(splits)
    x = activation_fn()(x)
    x = max_pool2d(pool_size=3, strides=2, padding='same')(x)

    # Stages
    c1, c2, c3 = channels[g]
    x = shufflenet_stage(layers=x, num_channels=c1, num_blocks=4)
    x = shufflenet_stage(layers=x, num_channels=c2, num_blocks=8)
    x = shufflenet_stage(layers=x, num_channels=c3, num_blocks=4)

    # Output
    x = conv2d(filters=last_channels, kernel_size=1)(x)
    x = activation_fn()(x)
    x = global_avg_pool2d()(x)
    return x

