"""A list of classes that wraps specifications dict for the ease of defining TensorforceAgents agents"""

from tensorforce import Agent

from typing import Optional, Union, List, Tuple, Dict, Callable
from agents.environment import SynchronousCARLAEnvironment

ListOrString = Optional[Union[str, List[str]]]
IntOrPair = Union[int, Tuple[int, int]]


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


class Layers:
    """Wraps some TensorForce's layers"""

    @staticmethod
    def layer_normalization() -> dict:
        return dict(type='keras', layer='LayerNormalization')

    @staticmethod
    def batch_normalization() -> dict:
        return dict(type='keras', layer='BatchNormalization')

    @staticmethod
    def gaussian_noise(stddev: float) -> dict:
        return dict(type='keras', layer='GaussianNoise', stddev=stddev)

    @staticmethod
    def spatial_dropout(rate: float) -> dict:
        return dict(type='keras', layer='SpatialDropout2D', rate=rate)

    @staticmethod
    def separable_conv2d(filters: int, kernel: IntOrPair, strides=(1, 1), padding='valid', depth_multiplier=1,
                         activation='relu', **kwargs) -> dict:
        return dict(type='keras', layer='SeparableConv2D', filters=filters, kernel_size=kernel, strides=strides,
                    depth_multiplier=depth_multiplier, activation=activation, padding=padding, **kwargs)

    @staticmethod
    def depthwise_conv2d(filters: int, kernel: IntOrPair, strides=(1, 1), padding='valid', depth_multiplier=1,
                         activation='relu', **kwargs) -> dict:
        return dict(type='keras', layer='DepthwiseConv2D', filters=filters, kernel_size=kernel, strides=strides,
                    depth_multiplier=depth_multiplier, activation=activation, padding=padding, **kwargs)

    @staticmethod
    def global_pool(reduction: str) -> dict:
        return dict(type='pooling', reduction=reduction)

    @staticmethod
    def global_avg_pooling() -> dict:
        return dict(type='pooling', reduction='mean')

    @staticmethod
    def flatten() -> dict:
        return dict(type='flatten')

    @staticmethod
    def dense(units: int, activation='relu', dropout=0.0) -> dict:
        return dict(type='dense', size=units, activation=activation, dropout=dropout)

    @staticmethod
    def conv2d_max_pool(filters, kernels: List[IntOrPair], strides: List[IntOrPair], activation='relu',
                        dropout=0.0, filter_increase=2) -> dict:
        assert len(kernels) == len(strides)
        layers = []

        for kernel, stride in zip(kernels, strides):
            layers.append(dict(type='conv2d', size=int(filters), window=kernel, stride=stride, activation=activation,
                               dropout=dropout))
            filters = filters * filter_increase

        layers.append(dict(type='pool2d', reduction='max', window=2, stride=2))

        return dict(type='block', layers=layers)


class NetworkSpec:
    """Ease the creation of a custom network specification; it also wraps layers as class methods."""

    def __init__(self, inputs: ListOrString = None):
        self.layers = []
        self.inputs(inputs)

    def add_layer(self, layer: dict):
        self.layers.append(layer)

    def add_normalization(self, kind: Optional[str] = None):
        if kind is None:
            return

        assert isinstance(kind, str)
        assert kind in ['batch', 'layer', 'instance', 'exponential']

        if kind == 'batch':
            self.layer_normalization()
        elif kind == 'layer':
            self.batch_normalization()
        elif kind == 'instance':
            self.instance_normalization()
        else:
            self.exponential_normalization()

    def inputs(self, inputs: ListOrString = None):
        """Input layer"""
        if inputs is None:
            return

        if isinstance(inputs, list):
            assert len(inputs) > 0
            self.add_layer(dict(type='retrieve', tensors=inputs))

        elif isinstance(inputs, str):
            self.add_layer(dict(type='retrieve', tensors=[inputs]))
        else:
            raise ValueError(f'Argument `inputs` should be of type `str` or `List[str]`, not `{type(inputs)}`.')

    def output(self, output: Optional[str] = None):
        """Output layer"""
        if output is None:
            return

        if isinstance(output, str):
            self.add_layer(dict(type='register', tensor=output))
        else:
            raise ValueError(f'Argument `output` should be of type `str`, not `{type(output)}`.')

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

    def conv2d(self, filters, kernel=3, stride=1, padding='same', activation='relu', dropout=0.0):
        self.add_layer(dict(type='conv2d', size=int(filters), window=kernel, stride=stride, padding=padding, bias=True,
                            activation=activation, dropout=dropout))

    def separable_conv2d(self, filters: int, kernel: IntOrPair, strides=(1, 1), padding='valid', depth_multiplier=1,
                         activation='relu', **kwargs):
        self.add_layer(dict(type='keras', layer='SeparableConv2D', filters=filters, kernel_size=kernel, strides=strides,
                            depth_multiplier=depth_multiplier, activation=activation, padding=padding, **kwargs))

    def depthwise_conv2d(self, filters: int, kernel: IntOrPair, strides=(1, 1), padding='valid', depth_multiplier=1,
                         activation='relu', **kwargs):
        self.add_layer(dict(type='keras', layer='DepthwiseConv2D', filters=filters, kernel_size=kernel, strides=strides,
                            depth_multiplier=depth_multiplier, activation=activation, padding=padding, **kwargs))

    def max_pool2d(self, window=2, stride=2, padding='same'):
        self.add_layer(dict(type='pool2d', reduction='max', window=window, stride=stride, padding=padding))

    def global_pooling(self, reduction: str):
        self.add_layer(dict(type='pooling', reduction=reduction))

    def global_avg_pooling(self):
        self.global_pooling(reduction='mean')

    def flatten(self):
        self.add_layer(dict(type='flatten'))

    def dense(self, units: int, activation='relu', dropout=0.0):
        self.add_layer(dict(type='dense', size=units, activation=activation, dropout=dropout))

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
    def input_layer(network: List[dict], inputs: ListOrString = None):
        if inputs is None:
            return

        if isinstance(inputs, list):
            assert len(inputs) > 0
            network.append(dict(type='retrieve', tensors=inputs))

        elif isinstance(inputs, str):
            network.append(dict(type='retrieve', tensors=[inputs]))
        else:
            raise ValueError(f'Argument `inputs` should be of type `str` or `List[str]`, not `{type(inputs)}`.')

    @staticmethod
    def output_layer(network: List[dict], output: Optional[str] = None):
        if output is None:
            return

        if isinstance(output, str):
            network.append(dict(type='register', tensor=output))
        else:
            raise ValueError(f'Argument `output` should be of type `str`, not `{type(output)}`.')

    @staticmethod
    def convolutional(inputs: ListOrString = None, output: str = None, initial_filters=32, kernel=(3, 3), pool='max',
                      activation='relu', stride=1, dropout=0.0, layers=2, normalization=None) -> List[dict]:
        network = []
        Networks.input_layer(network, inputs)

        for i in range(1, layers + 1):
            network.append(dict(type='conv2d', size=initial_filters * i, window=kernel, stride=stride,
                                activation=activation, dropout=dropout))

            if normalization == 'batch':
                network.append(Layers.batch_normalization())
            elif normalization == 'layer':
                network.append(Layers.layer_normalization())

            if pool:
                network.append(dict(type='pool2d', reduction=pool))

        network.append(dict(type='pooling', reduction='mean'))
        Networks.output_layer(network, output)

        return network

    @staticmethod
    def nvidia(inputs: str, output: str, filters=24, global_pool=False, units: List[int] = None,
               normalization='batch', activation='relu', dense_activation='relu') -> List[dict]:
        """Mimics the CNN described in the paper End-to-end Learning for Self-Driving Cars."""
        units = [1164, 100, 50, 10] if units is None else units

        # network architecture
        network = []
        Networks.input_layer(network, inputs)

        if normalization == 'batch':
            network.append(Layers.batch_normalization())
        elif normalization == 'layer':
            network.append(Layers.layer_normalization())

        network.append(Layers.conv2d_max_pool(filters * 1.0, kernels=[5], strides=[1], activation=activation))
        network.append(Layers.conv2d_max_pool(filters * 1.5, kernels=[5], strides=[1], activation=activation))

        network.append(Layers.conv2d_max_pool(filters * 2.0, kernels=[3], strides=[1], activation=activation))
        network.append(Layers.conv2d_max_pool(filters * 2.5, kernels=[3], strides=[1], activation=activation))

        if global_pool:
            network.append(Layers.global_avg_pooling())
        else:
            network.append(Layers.flatten())

        for neurons in units:
            network.append(Layers.dense(neurons, activation=dense_activation))

        Networks.output_layer(network, output)
        return network

    @staticmethod
    def dense(inputs: ListOrString = None, output: str = None, units=64, layers=2, activation='relu', dropout=0.0) \
            -> List[dict]:
        network = []
        Networks.input_layer(network, inputs)

        for i in range(layers):
            network.append(dict(type='dense', size=units, activation=activation, dropout=dropout))

        Networks.output_layer(network, output)
        return network

    @staticmethod
    def recurrent(inputs: ListOrString = None, output: str = None, embed: Optional[dict] = None, cell='gru', units=64,
                  return_final_state=True, activation='tanh', dropout=0.0) -> List[dict]:
        network = []
        Networks.input_layer(network, inputs)

        if isinstance(embed, dict):
            network.append(dict(type='embedding', size=embed['size'], num_embeddings=embed.get('num', None),
                                max_norm=embed.get('norm', None), bias=True, activation=embed.get('activation', 'tanh'),
                                dropout=dropout))

        network.append(dict(type='rnn', cell=cell, size=units, bias=True, activation=activation, dropout=dropout,
                            return_final_state=return_final_state))

        Networks.output_layer(network, output)
        return network

    @staticmethod
    def feature2d(inputs: str, output: str, shape: Tuple[int, int], filters: int, kernel=3, stride=1, layers=2,
                  activation='relu', dropout=0.0, global_pool='mean', normalization='layer') -> List[dict]:
        """A convolutional-like network to process matrix-like (2D) features.
            - If [global_pool=None] then flattening() is used instead.
            - If [normalization=None], no normalization is used at all.
        """
        assert len(shape) > 1

        network = []
        Networks.input_layer(network, inputs)
        network.append(dict(type='reshape', shape=shape + (1,)))  # make the shape like (h, w, 1)
        # network.append(dict(type='register', tensor='_reshape'))  # register reshape's output for later use

        # main conv. branch:
        for i in range(1, layers + 1):
            network.append(dict(type='conv2d', size=filters * i, window=kernel, stride=stride, activation=activation,
                                dropout=dropout))

            if normalization == 'batch':
                network.append(Layers.batch_normalization())
            elif normalization == 'layer':
                network.append(Layers.layer_normalization())

        if isinstance(global_pool, str):
            network.append(dict(type='pooling', reduction=global_pool))
        else:
            network.append(dict(type='flatten'))

        # # register main branch's output
        # network.append(dict(type='register', tensor="_main_out"))
        #
        # # summarize: apply a conv. kernel (1, shape[1]) with stride 1 and one filter
        # network.append(dict(type='retrieve', tensors=['_reshape']))
        # network.append(dict(type='conv2d', size=1, window=(1, shape[1]), stride=1, activation=activation))
        # network.append(dict(type='reshape', shape=(shape[0], )))
        # network.append(dict(type='register', tensor='_summary'))
        #
        # # concat main conv. branch output with summary output:
        # network.append(dict(type='retrieve', tensors=['_main_out', '_summary'], aggregation='concat'))

        Networks.output_layer(network, output)
        return network

    @staticmethod
    def complex(networks: [[dict]], layers=2, units=64, activation='relu', dropout=0.0, aggregation='concat',
                rnn: dict = None) -> List[dict]:
        network = networks
        outputs = []

        # find register (output) layers: expected at the end of each network.
        for net in networks:
            layer = net[-1]
            assert layer['type'] == 'register'

            outputs.append(layer['tensor'])

        # aggregate them
        network.append(dict(type='retrieve', tensors=outputs, aggregation=aggregation))

        for i in range(layers):
            network.append(dict(type='dense', size=units, activation=activation, dropout=dropout))

        if rnn and rnn.get('length', 0) > 0:
            network.append(dict(type='internal_rnn', cell=rnn.get('cell', 'lstm'), size=rnn.get('units', 128),
                                length=rnn.get('length'), bias=True, activation=rnn.get('activation', 'none'),
                                dropout=dropout))
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
    def policy(network: dict, distributions: str = None, temperature: Optional[float] = None, infer_states_value=False):
        policy_spec = dict(type='parametrized_distributions',
                           infer_states_value=infer_states_value,
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
    def saver():
        raise NotImplementedError

    @staticmethod
    def summarizer(directory='data/summaries', labels=None, frequency=100):
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
    def carla_agent(environment: SynchronousCARLAEnvironment, max_episode_timesteps: int, policy: dict,
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
    def agent_v2():
        # TODO: augment agent with an RNN
        # TODO: also stack 4 input (agent_v3?)
        # TODO: use separable-convolutions
        # TODO: reduce input size of image observation, e.g. 84x84, 75x105, 105x140, 150x200
        # TODO: use control instead of previous actions?
        pass

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
