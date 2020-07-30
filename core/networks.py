"""Networks architectures for CARLA Agent"""

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from typing import List, Dict, Union

from rl import networks


# def feature_net(input_layer: Input, noise=0.05, num_layers=2, units=32, activation='swish', dropout=0.5,
#                 name='feature_out') -> Layer:
#     """Network for vector features"""
#     x = GaussianNoise(stddev=noise)(input_layer)
#     x = tf.expand_dims(x, axis=0)
#
#     # recurrence
#     x, state = GRU(units, stateful=True, return_state=True)(x)
#     x = subtract([x, state])
#     x = LayerNormalization()(x)
#
#     # dense layers
#     for _ in range(num_layers):
#         x = Dense(units, activation=activation)(x)
#         x = Dropout(rate=dropout)(x)
#
#     return add([x, state], name=name)


# def convolutional(input_layer: Input, filters=32, filters_multiplier=1.0, blocks=2, units=64, dropout=0.2,
#                   kernel=(3, 3), strides=(2, 2), activation='swish', padding='same', name='conv_out'):
#     """Convolutional neural network"""
#     def block(prev_layer, i):
#         h = DepthwiseConv2D(kernel_size=kernel, padding=padding, activation=activation)(prev_layer)
#         h = Conv2D(filters=int(filters * i * filters_multiplier), kernel_size=kernel, padding=padding)(h)
#         h = ReLU(max_value=6.0, negative_slope=0.2)(h)
#         h = SpatialDropout2D(rate=dropout)(h)
#         h = MaxPooling2D(pool_size=3, strides=strides)(h)  # overlapping max-pool
#         return h
#
#     x = input_layer
#     k = 0
#     for _ in range(blocks):
#         x1 = block(x, i=k + 1)
#         x2 = DepthwiseConv2D(kernel_size=kernel, padding='same', activation='swish')(x1)
#         k += 2
#         x = concatenate([x1, x2], axis=-1)  # depth-concat
#         x = LayerNormalization()(x)
#
#     x = GlobalAveragePooling2D()(x)
#     x = LayerNormalization()(x)
#     x = tf.expand_dims(x, axis=0)
#
#     # 1-timestep RNN
#     x, state = GRU(units, stateful=True, return_state=True)(x)
#     return subtract([x, state], name=name)


# -----------------------------------------------------------------------------------------------------------

def feature_net(input_feature: Input, noise=0.05, num_layers=2, units=32, expansion=1.5, dropout=0.2) -> Layer:
    """Network for vector features"""
    def activation(layer: Layer):
        layer = BatchNormalization()(layer)
        layer = tf.nn.relu6(layer)
        return layer

    x = BatchNormalization()(input_feature)

    if noise > 0.0:
        x = GaussianNoise(stddev=noise)(x)

    # layers: linear dense -> dropout -> dense -> batch-norm
    for _ in range(num_layers):
        x = Dense(units=int(units * expansion), activation='linear')(x)

        if dropout > 0.0:
            x = Dropout(rate=dropout)(x)

        x = Dense(units, activation=None)(x)
        x = activation(x)

    return x


def embedding(layer: Layer, name: str, linear_units: int, units: int, activation=tf.nn.relu6, use_bias=True):
    if linear_units > 0:
        layer = Dense(units=linear_units, activation='linear')(layer)

    return Dense(units, activation=activation, use_bias=use_bias, name=name)(layer)


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


def residual_net(input_layer: Layer, initial_filters=32, kernel=3, depth_mul=1, dropout=0.25, activation=tf.nn.swish,
                 num_layers=4):
    def block(layer: Layer, filters: int, strides=1):
        h = SeparableConv2D(filters=filters, kernel_size=kernel, strides=strides,
                            depth_multiplier=depth_mul, padding='same')(layer)
        h = SpatialDropout2D(rate=dropout)(h)
        h = LayerNormalization()(h)
        return activation(h)

    def residual_block(layer: Layer, filters: int):
        h1 = block(layer, filters, strides=1)
        h1 = MaxPooling2D(pool_size=2)(h1)
        h2 = block(layer, filters, strides=2)

        return Add()([h1, h2])

    x = LayerNormalization()(input_layer)
    x = activation(x)

    for i in range(1, num_layers + 1):
        x = residual_block(x, filters=initial_filters * i)
        x = activation(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(rate=dropout)(x)
    x = LayerNormalization()(x)
    return activation(x)


# -------------------------------------------------------------------------------------------------
# -- ShuffleNet V2
# -------------------------------------------------------------------------------------------------

def shufflenet_v2(image_shape: tuple, num_features=40, g=1.0, leak=0.0, last_channels=1024, dilation=(1, 1),
                  strides=(2, 2)):
    """ShuffleNet-V2
       https://github.com/tensorpack/tensorpack/blob/master/examples/ImageNetModels/shufflenet.py
    """
    channels = {0.5: [48, 96, 192],
                1.0: [116, 232, 464],
                1.5: [176, 352, 704],
                2.0: [244, 488, 976]}

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
        layer = Conv2D(num_channels // 2, kernel_size=1, dilation_rate=dilation, padding='same')(layer)
        layer = activation(layer)
        layer = DepthwiseConv2D(kernel_size=3, strides=stride, dilation_rate=dilation, padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2D(num_channels - shortcut_channels, kernel_size=1, dilation_rate=dilation, padding='same')(layer)
        layer = activation(layer)

        if stride == 2:
            shortcut = DepthwiseConv2D(kernel_size=3, strides=2, dilation_rate=dilation, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)
            shortcut = Conv2D(shortcut_channels, kernel_size=1, dilation_rate=dilation, padding='same')(shortcut)
            shortcut = activation(shortcut)

        output = tf.concat([shortcut, layer], axis=-1)
        output = channel_shuffle(output)
        return output

    def shufflenet_stage(layer: Layer, num_channels: int, num_blocks: int):
        for i in range(num_blocks):
            layer = shufflenet_v2_unit(layer, num_channels, stride=strides if i == 0 else 1)
        return layer

    # Input
    inputs = Input(shape=image_shape, name='images')
    x = Conv2D(24, kernel_size=3, strides=strides)(inputs)
    x = activation(x)
    x = MaxPooling2D(pool_size=3, strides=strides, padding='same')(x)

    # Stages
    c1, c2, c3 = channels[g]
    x = shufflenet_stage(x, num_channels=c1, num_blocks=4)
    x = shufflenet_stage(x, num_channels=c2, num_blocks=8)
    x = shufflenet_stage(x, num_channels=c3, num_blocks=4)

    # Output
    x = Conv2D(last_channels, kernel_size=1)(x)
    x = activation(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_features, activation='sigmoid')(x)

    return Model(inputs, outputs, name='ShuffleNet-V2')


# -------------------------------------------------------------------------------------------------

class ContextLayer(Layer):
    def __init__(self, size: int):
        super().__init__()

        self.context_shape = (1, size)
        self.context = None
        self.reset()

        # z: how much new information to "preserve"
        self.linear = Dense(units=int(size * 1.5), activation='linear', name='context_linear')
        self.embed = Dense(units=size, activation=tf.nn.sigmoid, use_bias=False, name='context_preserve')

        # f: how much past information to "forget"
        self.forget = Dense(units=size, activation=tf.nn.sigmoid, use_bias=False, name='context_forget')

    # @tf.function
    # def compute_context(self, context, x):
    #     """Function used with tf.foldl, that computes (and thus updates) the context one element at a time"""
    #     # forget
    #     f = self.forget(context, axis=0)
    #     c = tf.multiply(f, context)
    #
    #     # new context
    #     return tf.add(c, x)

    @tf.function
    def compute_context(self, x):
        """Function used with tf.map_fn, that computes (and thus updates) the context one element at a time"""
        # forget
        f = self.forget(self.context)
        c = tf.multiply(f, self.context)

        # new context
        self.context.assign(tf.add(c, x))
        return self.context[0]

    # def call(self, inputs: list, training=False, **kwargs):
    #     # preserve
    #     z = concatenate(inputs)
    #     z = self.embed(self.linear(z))
    #
    #     # forget
    #     f = self.forget(self.context)
    #     c = multiply([f, self.context])
    #
    #     # new context
    #     self.context.assign(value=add(z, c)[-1])
    #     return self.context

    def call(self, inputs: list, training=False, **kwargs):
        # preserve
        z = concatenate(inputs)
        z = self.embed(self.linear(z))

        # self.context.assign(tf.foldl(fn=self.compute_context, elems=z, initializer=self.context))
        results = tf.map_fn(fn=self.compute_context, elems=z)
        return results

    def reset(self):
        self.context = tf.Variable(initial_value=tf.zeros(shape=self.context_shape), trainable=False)


# -------------------------------------------------------------------------------------------------

def dynamics_layers(inputs: dict, context_units: int, **kwargs):
    # process observation:
    image_out = networks.shufflenet_v2(inputs['state_image'], **kwargs.get('shufflenet', {}))
    road_out = feature_net(inputs['state_road'], **kwargs.get('road', {}),)
    vehicle_out = feature_net(inputs['state_vehicle'], **kwargs.get('vehicle', {}),)
    command_out = feature_net(inputs['state_command'], **kwargs.get('command', {}),)

    # embeddings (z):
    z_action = embedding(inputs['action'], **kwargs.get('action', {}), name='z_action')
    z_value = embedding(inputs['value'], **kwargs.get('value', {}), name='z_value')
    z_obs = Concatenate(name='z_obs')([image_out, road_out, vehicle_out, command_out])

    # context on actions, values, and observations:
    # #  z: how much new information to "preserve"
    # z = embedding(layer=concatenate([z_obs, z_action, z_value]), linear_units=int(context_units * 1.5),
    #               units=context_units, activation=tf.nn.sigmoid, use_bias=False, name='z')
    #
    # #  f: how much past information to "forget"
    # f = Dense(context_units, activation=tf.nn.sigmoid, use_bias=False)(inputs['context'])
    # c = multiply([f, inputs['context']])
    # context = Add(name='new_context')([z, c])

    context_layer = ContextLayer(size=context_units)
    context = context_layer([z_obs, z_action, z_value])

    # output:
    dynamics_out = Concatenate(name='dynamics_out')([context, z_obs, z_action, z_value])
    return dynamics_out, context_layer


def control_layers(inputs: dict, num_layers: int, units_multiplier: int, noise=0.0, dropout=0.0):
    assert units_multiplier > 0

    units = inputs['command'].shape[1] * units_multiplier
    x = feature_net(inputs['dynamics'], noise=noise, num_layers=num_layers, units=units, dropout=dropout)

    # implicit branching by 'command'
    command = tf.squeeze(inputs['command'])
    selector: tf.Tensor = tf.repeat(command, repeats=units_multiplier, axis=0)
    selector.set_shape((units,))

    branch = tf.boolean_mask(tensor=x, mask=selector, axis=1)
    branch.set_shape((None, units_multiplier))
    return branch


# -------------------------------------------------------------------------------------------------

class CARLANetwork(networks.PPONetwork):
    def __init__(self, agent, context_size: int,  control: dict, dynamics: dict):
        self.agent = agent  # workaround
        self.context_size = context_size
        self.context_layer = None
        self.intermediate_inputs = None

        self.dynamics = self.dynamics_model(**dynamics)
        self.memory = self.agent.memory

        super().__init__(agent, **control)

    def act(self, inputs: Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]]):
        raise NotImplementedError

    @tf.function
    def predict(self, inputs: Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]]):
        # inputs = inputs.copy()

        # 'last action' and 'last value' as inputs along the current observation
        if tf.shape(self.memory.actions)[0] == 0:
            inputs['action'] = tf.zeros(self.action_shape)
            inputs['value'] = tf.zeros((1, 1))
        else:
            inputs['action'] = self.memory.actions[-1]
            inputs['value'] = self.memory.values[-1]

        return super().predict(inputs)

    def policy_layers(self, inputs: Dict[str, Input], **kwargs) -> Layer:
        return control_layers(inputs, **kwargs)

    def dynamics_model(self, **kwargs) -> Model:
        """Implicit Dynamics Model,
           - Inputs: state/observation, action, value
        """
        inputs = self._get_input_layers()
        inputs['value'] = Input(shape=(1,), name='value')
        inputs['action'] = Input(shape=self.agent.action_shape, name='action')

        outputs, self.context_layer = dynamics_layers(inputs, context_units=self.context_size, **kwargs)

        self.intermediate_inputs = dict(dynamics=Input(shape=outputs.shape[1:], name='dynamics_int'),
                                        command=inputs['state_command'])

        return Model(inputs, outputs, name='Dynamics-Model')

    def policy_network(self, **kwargs) -> Model:
        last_layer = self.policy_layers(self.intermediate_inputs, **kwargs)
        action = self.get_distribution_layer(last_layer)

        return Model(inputs=self.intermediate_inputs, outputs=action, name='Policy-Network')

    def value_network(self, **kwargs) -> Model:
        branch = self.value_layers(self.intermediate_inputs, **kwargs)
        value = Dense(units=1, activation=None, dtype=tf.float32, name='value_head')(branch)

        return Model(inputs=self.intermediate_inputs, outputs=value, name='Value-Network')

    def reset(self):
        super().reset()
        self.context_layer.reset()

    def summary(self):
        super().summary()
        print('\n==== Dynamics Model ====')
        self.dynamics.summary()

    def save_weights(self):
        super().save_weights()
        self.dynamics.save_weights(filepath=self.agent.weights_path['dynamics'])

    def load_weights(self):
        super().load_weights()
        self.dynamics.load_weights(filepath=self.agent.weights_path['dynamics'], by_name=False)

# -------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    pass
