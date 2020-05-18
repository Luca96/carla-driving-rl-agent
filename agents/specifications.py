"""A list of classes that wraps specifications dict for the ease of defining TensorforceAgents agents"""

import tensorflow as tf

from tensorforce import Agent
from typing import Optional, Union, List, Tuple, Dict, Callable


ListOrString = Optional[Union[str, List[str]]]
IntOrPair = Union[int, Tuple[int, int]]
IntOrList = Union[int, List[int]]


class Objectives:
    """Specifications of TensorForce's objectives"""

    @staticmethod
    def deterministic_policy_gradient():
        return dict(type='det_policy_gradient')

    @staticmethod
    def plus(objective1: dict, objective2: dict):
        return dict(type='plus',
                    objective1=objective1,
                    objective2=objective2)

    @staticmethod
    def policy_gradient(ratio_based=False, clipping_value=0.0, early_reduce=True):
        return dict(type='policy_gradient',
                    ratio_based=ratio_based,
                    clipping_value=clipping_value,
                    early_reduce=early_reduce)

    @staticmethod
    def value(value='state', huber_loss=0.0, early_reduce=True):
        return dict(type='value',
                    value=value,
                    huber_loss=huber_loss,
                    early_reduce=early_reduce)


class Optimizers:
    """Specifications of TensorForce's optimizers."""

    @staticmethod
    def clipping_step(optimizer: dict, threshold: float, mode='global_norm'):
        return dict(type='clipping_step',
                    optimizer=optimizer,
                    threshold=threshold,
                    mode=mode)

    @staticmethod
    def evolutionary(learning_rate: float, num_samples=1, unroll_loop=False):
        return dict(type='evolutionary',
                    learning_rate=learning_rate,
                    num_samples=num_samples,
                    unroll_loop=unroll_loop)

    @staticmethod
    def multi_step(optimizer: dict, num_steps: int, unroll_loop=False):
        return dict(type='multi_step',
                    optimizer=optimizer,
                    num_steps=num_steps,
                    unroll_loop=unroll_loop)

    @staticmethod
    def natural_gradient(learning_rate: float, cg_max_iterations=10, cg_damping=0.001, cg_unroll_loop=False):
        return dict(type='natural_gradient',
                    learning_rate=learning_rate,
                    cg_max_iterations=cg_max_iterations,
                    cg_damping=cg_damping,
                    cg_unroll_loop=cg_unroll_loop)

    @staticmethod
    def optimizing_step(optimizer: dict, ls_max_iterations=10, ls_accept_ratio=0.9, ls_mode='exponential',
                        ls_parameter=0.5, ls_unroll_loop=False):
        return dict(type='optimizing_step',
                    optimizer=optimizer,
                    ls_max_iterations=ls_max_iterations,
                    ls_accept_ratio=ls_accept_ratio,
                    ls_mode=ls_mode,
                    ls_parameter=ls_parameter,
                    ls_unroll_loop=ls_unroll_loop)

    @staticmethod
    def plus(optimizer1: dict, optimizer2: dict):
        return dict(type='plus',
                    optimizer1=optimizer1,
                    optimizer2=optimizer2)

    @staticmethod
    def subsampling_step(optimizer: dict, fraction: float):
        return dict(type='subsampling_step',
                    optimizer=optimizer,
                    fraction=fraction)


# TODO: add **kwargs parameter
class NetworkSpec:
    """Ease the creation of a custom network specification; it also wraps layers as class methods."""

    def __init__(self, inputs: ListOrString = None, output: Optional[str] = None):
        self.layers: List[dict] = []
        self.inputs = None
        self.output = None

        if inputs is not None:
            self.set_inputs(inputs)

        if output is not None:
            self.set_output(output)

    def build(self) -> List[dict]:
        """Returns a list of layers specifications representing the network's architecture.
            - Note: both inputs and output must be not None!
        """
        if self.inputs is not None:
            assert self.output is not None
            layers = [self.inputs.copy()]
        else:
            assert self.output is None
            layers = []

        for layer in self.layers:
            layers.append(layer.copy())

        if self.output is not None:
            layers.append(self.output.copy())

        return layers

    def add_layer(self, layer: dict):
        self.layers.append(layer)

    def add_normalization(self, kind: Optional[str] = None):
        if kind is None:
            return

        assert isinstance(kind, str)
        assert kind in ['batch', 'layer', 'instance', 'exponential']

        if kind == 'batch':
            self.batch_normalization()
        elif kind == 'layer':
            self.layer_normalization()
        elif kind == 'instance':
            self.instance_normalization()
        else:
            self.exponential_normalization()

    def set_inputs(self, inputs: ListOrString, aggregation='concat'):
        """Input layer"""
        if isinstance(inputs, list):
            assert len(inputs) > 0
            self.inputs = dict(type='retrieve', tensors=inputs, aggregation=aggregation)

        elif isinstance(inputs, str):
            # self.inputs = dict(type='retrieve', tensors=[inputs], aggregation=aggregation)
            self.inputs = dict(type='retrieve', tensors=inputs, aggregation=aggregation)
        else:
            raise ValueError(f'Argument `inputs` should be of type `str` or `List[str]`, not `{type(inputs)}`.')

    def set_output(self, output: Optional[str]):
        """Output layer"""
        if isinstance(output, str):
            self.output = dict(type='register', tensor=output)
        else:
            raise ValueError(f'Argument `output` should be of type `str`, not `{type(output)}`.')

    def reshape(self, shape, **kwargs):
        self.add_layer(dict(type='reshape', shape=shape, **kwargs))

    def layer_normalization(self):
        self.add_layer(dict(type='keras', layer='LayerNormalization'))

    def batch_normalization(self):
        self.add_layer(dict(type='keras', layer='BatchNormalization'))

    def exponential_normalization(self, decay=0.999):
        self.add_layer(dict(type='exponential_normalization', decay=decay))

    def instance_normalization(self):
        self.add_layer(dict(type='instance_normalization'))

    def gaussian_noise(self, stddev: float):
        self.add_layer(dict(type='keras', layer='GaussianNoise', stddev=stddev))

    def spatial_dropout(self, rate: float):
        self.add_layer(dict(type='keras', layer='SpatialDropout2D', rate=rate))

    def conv2d(self, filters, kernel: IntOrPair = 3, stride: IntOrPair = 1, activation='relu', dropout=0.0,
               padding='same'):
        self.add_layer(dict(type='conv2d', size=int(filters), window=kernel, stride=stride, padding=padding, bias=True,
                            activation=activation, dropout=dropout))

    def separable_conv2d(self, filters: int, kernel: IntOrPair, strides=(1, 1), padding='valid', depth_multiplier=1,
                         activation='relu', **kwargs):
        self.add_layer(dict(type='keras', layer='SeparableConv2D', filters=filters, kernel_size=kernel, strides=strides,
                            depth_multiplier=depth_multiplier, activation=activation, padding=padding, **kwargs))

    def depthwise_conv2d(self, kernel: IntOrPair, strides=(1, 1), padding='valid', depth_multiplier=1,
                         activation='relu', **kwargs):
        self.add_layer(dict(type='keras', layer='DepthwiseConv2D', kernel_size=kernel, strides=strides,
                            depth_multiplier=depth_multiplier, activation=activation, padding=padding, **kwargs))

    def pool2d(self, reduction: str, window: int, stride: int, padding='same'):
        self.add_layer(dict(type='pool2d', reduction=reduction, window=window, stride=stride, padding=padding))

    def max_pool2d(self, window=2, stride=2, padding='same'):
        self.pool2d(reduction='max', window=window, stride=stride, padding=padding)

    def global_pooling(self, reduction: str, **kwargs):
        self.add_layer(dict(type='pooling', reduction=reduction, **kwargs))

    def global_avg_pooling(self, **kwargs):
        self.global_pooling(reduction='mean', **kwargs)

    def flatten(self, **kwargs):
        self.add_layer(dict(type='flatten', **kwargs))

    def function(self, tf_function, **kwargs):
        assert callable(tf_function)
        self.add_layer(dict(type='function', function=tf_function, **kwargs))

    def dense(self, units: int, activation='relu', dropout=0.0):
        self.add_layer(dict(type='dense', size=units, activation=activation, dropout=dropout))

    def embedding(self, size: int, num_embeddings=None, max_norm=None, activation='tanh', dropout=0.0):
        self.add_layer(dict(type='embedding', size=size, num_embeddings=num_embeddings, max_norm=max_norm,
                            activation=activation, bias=True, dropout=dropout))

    def rnn(self, size: int, cell='gru', return_final_state=True, activation='tanh', dropout=0.0, **kwargs):
        self.add_layer(dict(type='rnn', cell=cell, size=size, return_final_state=return_final_state,
                            activation=activation, dropout=dropout, **kwargs))

    def keras_rnn(self, units: int, cell='gru', activation='tanh', recurrent_activation='sigmoid', dropout=0.0,
                  recurrent_dropout=0.0, return_sequences=False, return_state=False, stateful=False):
        """
        - return_sequences: Whether to return the last output in the output sequence, or the full sequence.
        - return_state: Whether to return the last state in addition to the output. (suggested: True)
        - stateful: If True, the last state for each sample at index i in a batch will be used as initial state for the
                    sample of index i in the following batch. (suggested: True)

        """
        assert cell in ['gru', 'lstm']

        self.add_layer(dict(type='keras', layer=cell.upper(), units=units, activation=activation, dropout=dropout,
                            recurrent_activation=recurrent_activation, recurrent_dropout=recurrent_dropout,
                            return_sequences=return_sequences, return_state=return_state, stateful=stateful))

    def relu6(self):
        self.function(tf_function=tf.nn.relu6)

    def leaky_relu(self):
        self.function(tf_function=tf.nn.leaky_relu)

    def conv2d_max_pool(self, filters, kernels: List[IntOrPair], strides: List[IntOrPair], activation='relu',
                        dropout=0.0, filter_increase=2):
        assert len(kernels) == len(strides)

        for kernel, stride in zip(kernels, strides):
            self.conv2d(filters, kernel, stride, activation=activation, dropout=dropout)
            filters = filters * filter_increase

        self.max_pool2d()


class Networks:
    @staticmethod
    def auto(size=64, depth=2, final_size=None, final_depth=1, internal_rnn=False):
        return dict(type='auto',
                    size=size,
                    depth=depth,
                    final_size=final_size,
                    final_depth=final_depth,
                    internal_rnn=internal_rnn)

    @staticmethod
    def convolutional(inputs: ListOrString = None, output: str = None, initial_filters=32, kernel=(3, 3), pool='max',
                      activation='relu', stride=1, dropout=0.0, layers=2, normalization=None) -> List[dict]:
        network = NetworkSpec()
        network.set_inputs(inputs)

        for i in range(1, layers + 1):
            network.conv2d(initial_filters * i, kernel, stride, activation, dropout)
            network.add_normalization(kind=normalization)

            if isinstance(pool, str):
                network.pool2d(reduction=pool, window=2, stride=2)

        network.global_avg_pooling()
        network.set_output(output)
        return network.build()

    @staticmethod
    def my_cnn(inputs: str, output: str, filters=32, activation1='tanh', activation2='elu', dropout=0.2, reshape=None,
               layers=(2, 5), noise=0.05, middle_noise=False, normalization='instance', middle_normalization=False,
               filters_multiplier=1, final_dense: dict = None):
        assert len(layers) == 2
        assert filters_multiplier > 0
        network = NetworkSpec(inputs, output)

        if isinstance(reshape, tuple):
            network.reshape(shape=reshape)

        network.add_normalization(kind=normalization)

        if noise > 0.0:
            network.gaussian_noise(stddev=noise)

        # part 1:
        for i in range(1, layers[0] + 1):
            network.depthwise_conv2d(kernel=(3, 3), padding='same', depth_multiplier=1, activation=activation1)
            network.conv2d(filters=int(filters * i * filters_multiplier), kernel=(3, 3), padding='same',
                           activation=activation2)
            network.spatial_dropout(rate=dropout)
            network.max_pool2d(window=3, stride=2)  # overlapping max-pool

        if middle_normalization:
            network.add_normalization(kind=normalization)

        if middle_noise:
            network.gaussian_noise(stddev=noise)

        # part 2:
        for i in range(1, layers[1] + 1):
            padding = 'same' if i % 2 == 0 else 'valid'
            network.depthwise_conv2d(kernel=(3, 3), padding=padding, depth_multiplier=1, activation=activation1)
            network.conv2d(filters=int(filters * (i + layers[0]) * filters_multiplier), kernel=(3, 3), padding='valid',
                           activation=activation2)
            network.spatial_dropout(rate=dropout)

        network.global_avg_pooling()

        # dense layers at the end:
        final_dense = final_dense if isinstance(final_dense, dict) else dict()
        assert ('units' in final_dense) and ('layers' in final_dense)

        for _ in range(final_dense['layers']):
            network.dense(units=final_dense['units'], activation=final_dense.get('activation', 'relu'),
                          dropout=final_dense.get('dropout', 0.0))

        return network.build()

    @staticmethod
    def nvidia(inputs: str, output: str, filters=24, global_pool=False, units=[1164, 100, 50, 10],
               normalization='batch', activation='relu', dropout=0.0) -> List[dict]:
        """Mimics the CNN described in the paper End-to-end Learning for Self-Driving Cars."""
        network = NetworkSpec()
        network.set_inputs(inputs)
        network.add_normalization(kind=normalization)

        # block 1: conv k5 -> max pool -> conv k5 -> max pool
        network.conv2d_max_pool(filters, kernels=[5], strides=[1], activation=activation, dropout=dropout)
        network.conv2d_max_pool(filters, kernels=[5], strides=[1], activation=activation, dropout=dropout)

        # block 2: conv k3 -> max pool -> conv k3 -> max pool
        network.conv2d_max_pool(filters, kernels=[3], strides=[1], activation=activation, dropout=dropout)
        network.conv2d_max_pool(filters, kernels=[3], strides=[1], activation=activation, dropout=dropout)

        if global_pool:
            network.global_avg_pooling()
        else:
            network.flatten()

        if len(units) > 0:
            network.dense(units[0], activation='relu', dropout=dropout)

            for units in units[1:]:
                network.dense(units, activation='tanh', dropout=dropout)

        network.set_output(output)
        return network.build()

    @staticmethod
    def dense(inputs: ListOrString = None, output: str = None, units: IntOrList = 64, layers: Optional[int] = 2,
              activation='relu', dropout=0.0, embed=None) -> List[dict]:
        network = NetworkSpec(inputs, output)

        if isinstance(embed, int):
            network.embedding(size=embed)

        if isinstance(units, int):
            assert isinstance(layers, int) and layers > 0
            units = [units] * layers
        else:
            assert isinstance(units, list) and len(units) > 0

        for neurons in units:
            network.dense(neurons, activation, dropout)

        return network.build()

    @staticmethod
    def recurrent(inputs: ListOrString = None, output: str = None, embed: Optional[dict] = None, cell='gru', units=64,
                  return_final_state=True, activation='tanh', dropout=0.0) -> List[dict]:
        network = NetworkSpec()
        network.set_inputs(inputs)

        if isinstance(embed, dict):
            network.embedding(size=embed['size'], num_embeddings=embed.get('num', None), dropout=dropout,
                              max_norm=embed.get('norm', None), activation=embed.get('activation', 'tanh'))

        network.rnn(units, cell, return_final_state, activation, dropout)
        network.set_output(output)

        return network.build()

    @staticmethod
    def feature2d(inputs: str, output: str, shape: Tuple[int, int], filters: int, kernel=3, stride=1, layers=2,
                  activation='swish', dropout=0.0, global_pool='mean', normalization='layer') -> List[dict]:
        """A convolutional-like network to process matrix-like (2D) features.
            - If [global_pool=None] then flattening() is used instead.
            - If [normalization=None], no normalization is used at all.
        """
        assert len(shape) > 1

        network = NetworkSpec(inputs, output)
        network.add_normalization(kind=normalization)
        network.reshape(shape=shape + (1,))  # makes the shape like (h, w, 1)

        # main conv. branch:
        for i in range(1, layers + 1):
            network.conv2d(filters * i, kernel, stride, activation, dropout)

        if isinstance(global_pool, str):
            network.global_pooling(reduction=global_pool)
        else:
            network.flatten()

        return network.build()

    @staticmethod
    def feature2d_v2(inputs: str, output: str, shape: Tuple[int, int], filters: int, layers=2, dense: dict = None,
                     activation1='tanh', activation2='elu', spatial_dropout=0.2, normalization='layer',
                     depth_multiplier=2) -> List[dict]:
        """A convolutional-like network to process matrix-like (2D) features.
            - If [normalization=None], no normalization is used at all.
        """
        assert len(shape) > 1
        assert 0.0 <= spatial_dropout < 1.0

        network = NetworkSpec(inputs, output)
        network.add_normalization(kind=normalization)
        network.reshape(shape=shape + (1,))

        # convolutional embedding (transform input with pointwise, i.e. 1x1, convolutions)
        network.conv2d(filters, kernel=1, activation=activation1, dropout=0.0)

        for i in range(1, layers + 1):
            filters *= depth_multiplier
            network.separable_conv2d(filters=round(filters), kernel=1, activation=activation2)

            if spatial_dropout > 0.0:
                network.spatial_dropout(rate=spatial_dropout)

        # reduce depth to 1 (convolutional aggregation)
        network.separable_conv2d(filters=1, kernel=1, activation=activation1)
        network.flatten()

        # dense bottleneck
        if isinstance(dense, dict):
            for _ in range(dense['layers']):
                network.dense(units=dense['units'], activation=dense.get('activation', 'relu'),
                              dropout=dense.get('dropout', 0.0))

        return network.build()

    @staticmethod
    def feature2d_skip(inputs: str, output: str, name: str, shape: Tuple[int, int], filters: int, kernel=3, stride=1,
                       layers=2, activation='swish', dropout=0.0, global_pool='mean',
                       normalization='layer') -> List[dict]:
        """A convolutional-like network to process matrix-like (2D) features.
            - If [global_pool=None] then flattening() is used instead.
            - If [normalization=None], no normalization is used at all.
        """
        assert len(shape) > 1
        network = NetworkSpec(inputs, output)
        network.add_normalization(kind=normalization)
        network.reshape(shape=shape + (1,))
        network.add_layer(dict(type='register', tensor=f'reshape_{name}_out'))
        network.add_layer(dict(type='retrieve', tensors=f'reshape_{name}_out'))

        # main conv. branch:
        for i in range(1, layers + 1):
            network.conv2d(filters * i, kernel=kernel, stride=stride, activation=activation, dropout=dropout,
                           padding='valid')

        if isinstance(global_pool, str):
            network.global_pooling(reduction=global_pool)
        else:
            network.flatten()

        # register main branch's output
        network.add_layer(dict(type='register', tensor=f"conv_{name}_out"))

        # summarize: apply a conv. kernel (1, shape[1]) with stride 1 and one filter
        network.add_layer(dict(type='retrieve', tensors=f'reshape_{name}_out'))

        network.conv2d(filters=1, kernel=(1, shape[1]), stride=1, padding='valid', activation=activation)

        network.reshape(shape=(shape[0],))
        network.add_layer(dict(type='register', tensor=f'avg_{name}_out'))

        # concat main conv. branch output with summary output:
        network.add_layer(dict(type='retrieve', tensors=[f'conv_{name}_out', f'avg_{name}_out'], aggregation='concat'))

        return network.build()

    @staticmethod
    def complex(networks: [[dict]], layers=2, units: IntOrList = 64, activation='relu', dropout=0.0,
                aggregation='concat', rnn: dict = None, discretize=None) -> List[dict]:
        network = networks
        outputs = []

        # find register (output) layers: expected at the end of each network.
        for net in networks:
            layer = net[-1]
            assert layer['type'] == 'register'

            outputs.append(layer['tensor'])

        # aggregate them
        network.append(dict(type='retrieve', tensors=outputs, aggregation=aggregation))

        network.extend(Networks.dense(units=units, layers=layers, activation=activation, dropout=dropout))

        if rnn and rnn.get('length', 0) > 0:
            network.append(dict(type='internal_rnn', cell=rnn.get('cell', 'lstm'), size=rnn.get('units', 128),
                                length=rnn.get('length'), bias=True, activation=rnn.get('activation', 'none'),
                                dropout=dropout))

        if isinstance(discretize, int) and discretize >= 0:
            def tf_discrete(x):
                multiplier = tf.constant(10**discretize, dtype=x.dtype)
                return tf.round(x * multiplier) / multiplier

            network.append(dict(type='function', function=tf_discrete))

        return network


class Specifications:
    """Explicits TensorForce's specifications as dicts"""
    objectives = Objectives
    optimizers = Optimizers
    networks = Networks

    # Short names:
    obj = objectives
    opt = optimizers
    net = networks

    @staticmethod
    def update(unit: str, batch_size: int, frequency=None, start: int = None):
        update_spec = dict(unit=unit, batch_size=batch_size)

        if frequency is not None:
            update_spec['frequency'] = frequency

        if start is not None:
            update_spec['start'] = start

        return update_spec

    @staticmethod
    def reward_estimation(horizon: int, discount=1.0, estimate_horizon=False, estimate_actions=False,
                          estimate_advantage=False):
        return dict(horizon=horizon,
                    discount=discount,
                    estimate_horizon=estimate_horizon,
                    estimate_actions=estimate_actions,
                    estimate_advantage=estimate_advantage)

    @staticmethod
    def policy(network: Union[dict, List[dict]], distributions: str = None, temperature: Optional[float] = None,
               infer_states_value=False, use_beta_distribution=True):
        policy_spec = dict(type='parametrized_distributions',
                           infer_states_value=infer_states_value,
                           use_beta_distribution=use_beta_distribution,
                           network=network)

        if distributions is not None:
            policy_spec['distributions'] = dict(type=distributions)

        if temperature is not None:
            policy_spec['temperature'] = temperature

        return policy_spec

    @staticmethod
    def network_v0(dropout=0.2):
        return Networks.complex(networks=[
            Networks.convolutional(inputs='image', activation='relu', layers=5, stride=2, initial_filters=32, pool=None,
                                   dropout=dropout, output='image_out'),
            Networks.dense(inputs='vehicle_features', layers=2, units=32, dropout=dropout,
                           output='vehicle_out'),
            Networks.dense(inputs='road_features', layers=2, units=24, dropout=dropout,
                           output='road_out'),
            Networks.dense(inputs='previous_actions', layers=1, units=16, dropout=dropout,
                           output='actions_out')],

            layers=2,
            activation='relu',
            rnn=None,
            units=200)

    @staticmethod
    def network_v1(conv: dict, rnn: dict = None, final: dict = None, dropout=0.2):
        final = final or dict()

        return Networks.complex(networks=[
            Networks.convolutional(inputs='image',
                                   activation=conv.get('activation', 'relu'),
                                   layers=conv.get('layers', 4),
                                   stride=conv.get('stride', 2),
                                   initial_filters=conv.get('filters', 32),
                                   pool=conv.get('pooling', None),
                                   dropout=dropout,
                                   output='image_out'),
            Networks.dense(inputs='vehicle_features', layers=2, units=32, dropout=dropout,
                           output='vehicle_out'),
            Networks.dense(inputs='road_features', layers=2, units=24, dropout=dropout,
                           output='road_out'),
            Networks.dense(inputs='previous_actions', layers=2, units=16, dropout=dropout,
                           output='actions_out')],

            layers=final.get('layers', 2),
            activation=final.get('activation', 'none'),
            rnn=rnn,
            units=final.get('units', 256))

    @staticmethod
    def network_v2(conv: dict, rnn: dict = None, final: dict = None, dropout=0.2):
        final = final or dict()

        return Networks.complex(networks=[
            Networks.convolutional(inputs='image',
                                   activation=conv.get('activation', 'relu'),
                                   layers=conv.get('layers', 4),
                                   stride=conv.get('stride', 2),
                                   initial_filters=conv.get('filters', 32),
                                   pool=conv.get('pooling', None),
                                   dropout=dropout,
                                   output='image_out'),

            [dict(type='retrieve', tensors=['vehicle_features', 'road_features', 'previous_actions']),
             dict(type='register', tensor='features_out')]],

            layers=final.get('layers', 2),
            activation=final.get('activation', 'none'),
            rnn=rnn,
            units=final.get('units', 256))

    @staticmethod
    def network_v3(features: Dict[str, dict], conv: dict = None, final: dict = None, dropout=0.2):
        assert isinstance(features, dict)
        conv = conv or dict()
        final = final or dict()

        networks = [Networks.convolutional(inputs='image', output='image_out', layers=conv.get('layers', 4),
                                           activation=conv.get('activation', 'leaky-relu'),
                                           stride=conv.get('stride', 2), initial_filters=conv.get('filters', 32),
                                           pool=conv.get('pooling', None),
                                           normalization=conv.get('normalization', None), dropout=dropout)]

        for feature_name, args in features.items():
            feature_net = Networks.feature2d(inputs=feature_name, output=feature_name + "_out", shape=args['shape'],
                                             filters=args['filters'], kernel=args.get('kernel', 3),
                                             stride=args.get('stride', 1), layers=args.get('layers', 2),
                                             activation=args.get('activation', 'relu'), dropout=dropout,
                                             global_pool=args.get('global_pool', 'mean'),
                                             normalization=args.get('normalization', 'layer'))
            networks.append(feature_net)

        # network for 'past_skill'
        networks.append(Networks.dense(inputs='past_skills', output='past_skills_out', units=32, layers=3,
                                       activation='tanh', dropout=dropout))

        return Networks.complex(networks=networks, layers=final.get('layers', 2),
                                activation=final.get('activation', 'none'), units=final.get('units', 256))

    @staticmethod
    def network_v4(convolutional: Dict[str, dict], features: Dict[str, dict], dense: Dict[str, dict],
                   final: dict = None) -> List[dict]:
        assert isinstance(convolutional, dict)
        assert isinstance(features, dict)
        assert isinstance(dense, dict)
        final = final or dict()
        networks = []

        for name, args in convolutional.items():
            conv_net = Networks.my_cnn(inputs=name, output=name + '_out', **args)
            networks.append(conv_net)

        for name, args in features.items():
            feature_net = Networks.feature2d(inputs=name, output=name + "_out", **args)
            networks.append(feature_net)

        for name, args in dense.items():
            dense_net = Networks.dense(inputs=name, output=name + '_out', **args)
            networks.append(dense_net)

        return Networks.complex(networks=networks, **final)

    @staticmethod
    def network_v5(convolutional: Dict[str, dict], features: Dict[str, dict], dense: Dict[str, dict],
                   final: dict = None) -> List[dict]:
        assert isinstance(convolutional, dict)
        assert isinstance(features, dict)
        assert isinstance(dense, dict)
        final = final or dict()
        networks = []

        for name, args in convolutional.items():
            conv_net = Networks.my_cnn(inputs=name, output=name + '_out', **args)
            networks.append(conv_net)

        for name, args in features.items():
            feature_net = Networks.feature2d_v2(inputs=name, output=name + "_out", **args)
            networks.append(feature_net)

        for name, args in dense.items():
            dense_net = Networks.dense(inputs=name, output=name + '_out', **args)
            networks.append(dense_net)

        return Networks.complex(networks=networks, **final)

    @staticmethod
    def rnn_network(conv: dict = None, rnn: dict = None, final: dict = None, dropout=0.2):
        conv = conv or dict()
        final = final or dict()

        return Networks.complex(networks=[
            Networks.convolutional(inputs='image',
                                   activation=conv.get('activation', 'leaky-relu'),
                                   layers=conv.get('layers', 4),
                                   stride=conv.get('stride', 2),
                                   initial_filters=conv.get('filters', 32),
                                   pool=conv.get('pooling', None),
                                   normalization=conv.get('normalization', None),
                                   dropout=dropout,
                                   output='image_out'),

            Networks.recurrent(inputs='radar', units=12, dropout=dropout, output='radar_out'),
            Networks.recurrent(inputs='vehicle', units=34, dropout=dropout, output='vehicle_out'),
            Networks.recurrent(inputs='road', units=20, dropout=dropout, output='road_out'),
            Networks.recurrent(inputs='previous', units=14, dropout=dropout, output='actions_out')],

            layers=final.get('layers', 2),
            activation=final.get('activation', 'none'),
            rnn=rnn,
            units=final.get('units', 256))

    @staticmethod
    def saver(directory: str, filename: str, frequency=600, load=True) -> dict:
        return dict(directory=directory, filename=filename, frequency=frequency, load=load)

    @staticmethod
    def summarizer(directory='data/summaries', labels=None, frequency=100) -> dict:
        # ['graph', 'entropy', 'kl-divergence', 'losses', 'rewards'],
        return dict(directory=directory,
                    labels=labels or ['entropy', 'action-entropies', 'gaussian', 'exploration', 'beta',
                                      'kl-divergences', 'losses', 'rewards'],
                    frequency=frequency)

    @staticmethod
    def exp_decay(steps: int, rate: float, unit='timesteps', initial_value=1.0, increasing=False, staircase=False):
        return dict(type='decaying',
                    decay='exponential',
                    unit=unit,
                    initial_value=initial_value,
                    increasing=increasing,
                    staircase=staircase,
                    decay_steps=steps,
                    decay_rate=rate)

    @staticmethod
    def linear_decay(initial_value: float, final_value: float, steps: int, unit='updates', cycle=False):
        return dict(type='decaying', decay='polynomial', power=1.0, unit=unit, decay_steps=steps, increasing=False,
                    initial_value=initial_value, final_value=final_value, cycle=cycle)

    @staticmethod
    def carla_agent(environment, max_episode_timesteps: int, policy: dict,
                    critic: Optional[dict], discount=1.0, horizon=100, batch_size=256, update_frequency=64, **kwargs):
        if critic is None:
            critic_policy = None
            critic_opt = None
            critic_obj = None
        else:
            critic_policy = Specifications.policy(distributions=critic.get('distributions', 'gaussian'),
                                                  network=critic['network'],
                                                  temperature=critic.get('temperature', 1.0))
            critic_opt = critic.get('optimizer', dict(type='adam', learning_rate=3e-4))
            critic_obj = critic.get('objective', Objectives.value(value='state', huber_loss=0.1, early_reduce=True))

        return Agent.create(agent='tensorforce',
                            environment=environment,
                            max_episode_timesteps=max_episode_timesteps,

                            update=Specifications.update(unit='timesteps', batch_size=batch_size,
                                                         frequency=update_frequency,
                                                         start=batch_size),
                            # Policy
                            policy=Specifications.policy(network=policy['network'],
                                                         distributions=policy.get('distributions', None),
                                                         temperature=policy.get('temperature', 1.0),
                                                         infer_states_value=True),
                            memory=dict(type='recent'),
                            optimizer=policy.get('optimizer', dict(type='adam', learning_rate=3e-4)),
                            objective=Objectives.policy_gradient(clipping_value=0.2, early_reduce=True),

                            # Critic
                            baseline_policy=critic_policy,
                            baseline_optimizer=critic_opt,
                            baseline_objective=critic_obj,

                            # Reward
                            reward_estimation=dict(discount=discount,
                                                   horizon=horizon,
                                                   estimate_horizon='early',
                                                   estimate_advantage=True),
                            **kwargs)

    @staticmethod
    def my_preprocessing(image_shape=(105, 140, 1), normalization=False, stack_images=4):
        img_prep = [dict(type='image', width=image_shape[1], height=image_shape[0], grayscale=image_shape[2] == 1)]

        if normalization:
            img_prep.append(dict(type='instance_normalization'))

        if stack_images > 0:
            img_prep.append(dict(type='sequence', length=stack_images, axis=-1, concatenate=True))  # depth concat

        return dict(image=img_prep,
                    vehicle_features=dict(type='deltafier'),
                    road_features=dict(type='deltafier'))

    @staticmethod
    def sequence_preprocessing(image_shape=(105, 140, 1), time_horizon=10):
        assert time_horizon > 0

        return dict(image=[dict(type='image', width=image_shape[1], height=image_shape[0],
                                grayscale=image_shape[2] == 1),
                           dict(type='sequence', length=time_horizon, axis=-1)],

                    vehicle_features=dict(type='sequence', length=time_horizon, axis=0, concatenate=False),
                    road_features=dict(type='sequence', length=time_horizon, axis=0, concatenate=False),
                    previous_actions=dict(type='sequence', length=time_horizon, axis=0, concatenate=False))
