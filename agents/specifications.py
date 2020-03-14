"""A list of classes that wraps specifications dict for the ease of defining TensorforceAgents agents"""


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
    def policy_gradient(ratio_based=False, clipping_value=0.0, early_reduce=False):
        return dict(type='policy_gradient',
                    ratio_based=ratio_based,
                    clipping_value=clipping_value,
                    early_reduce=early_reduce)

    @staticmethod
    def value(value='state', huber_loss=0.0, early_reduce=False):
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


class Specifications:
    """Explicits TensorForce's specifications as dicts"""
    objectives = Objectives
    optimizers = Optimizers

    # Short names:
    obj = objectives
    opt = optimizers

    @staticmethod
    def update(unit: str, batch_size: int, frequency=None, start: int = None):
        return dict(unit=unit,
                    batch_size=batch_size,
                    frequency=frequency if frequency else batch_size,
                    start=start if start else batch_size)

    @staticmethod
    def reward_estimation(horizon: int, discount=1.0, estimate_horizon=False, estimate_actions=False,
                          estimate_advantage=False):
        return dict(horizon=horizon,
                    discount=discount,
                    estimate_horizon=estimate_horizon,
                    estimate_actions=estimate_actions,
                    estimate_advantage=estimate_advantage)

    @staticmethod
    def auto_network(size=64, depth=2, final_size=None, final_depth=1, internal_rnn=False):
        return dict(type='auto',
                    size=size,
                    depth=depth,
                    final_size=final_size,
                    final_depth=final_depth,
                    internal_rnn=internal_rnn)

    @staticmethod
    def conv_network(inputs: [str] = None, output: str = None, initial_filters=32, kernel=(3, 3), pool='max',
                     activation='relu', stride=1, dilation=1, dropout=0.0, layers=2, normalization='instance'):
        network = []

        if inputs is not None:
            if isinstance(inputs, list) and len(inputs) > 0:
                network.append(dict(type='retrieve', tensors=inputs))
            elif isinstance(inputs, str):
                network.append(dict(type='retrieve', tensors=[inputs]))

        for i in range(1, layers + 1):
            filters = initial_filters * i

            if stride > 1:
                convolution = dict(type='conv2d', size=filters, window=kernel, stride=stride, activation=activation,
                                   dropout=dropout)
            else:
                convolution = dict(type='conv2d', size=filters, window=kernel, dilation=dilation, activation=activation,
                                   dropout=dropout)

            network.append(convolution)

            if normalization == 'instance':
                network.append(dict(type='instance_normalization'))
            elif normalization == 'exponential' or normalization == 'exp':
                network.append(dict(type='exponential_normalization'))

            if pool is not None:
                network.append(dict(type='pool2d', reduction=pool))

        network.append(dict(type='pooling', reduction='mean'))

        if output is not None:
            network.append(dict(type='register', tensor=output))

        return network

    @staticmethod
    def dense_network(inputs: [str] = None, output: str = None, units=64, layers=2, activation='relu', dropout=0.0,
                      normalization='instance'):
        network = []

        if inputs is not None:
            if isinstance(inputs, list) and len(inputs) > 0:
                network.append(dict(type='retrieve', tensors=inputs))
            elif isinstance(inputs, str):
                network.append(dict(type='retrieve', tensors=[inputs]))

        for i in range(layers):
            network.append(dict(type='dense', size=units, activation=activation, dropout=dropout))

            if normalization == 'instance':
                network.append(dict(type='instance_normalization'))
            elif normalization == 'exponential' or normalization == 'exp':
                network.append(dict(type='exponential_normalization'))

        if output is not None:
            network.append(dict(type='register', tensor=output))

        return network

    @staticmethod
    def complex_network_old(networks: [[dict]], layers=2, units=64, activation='relu', dropout=0.0, aggregation='concat'):
        network = []
        tensors = []

        for net in networks:
            # copy layers
            network.extend(net)

            # get registered layers
            # layer = net[-1]
            layer = net[1]
            assert layer['type'] == 'register'

            tensors.append(layer['tensor'])

        # aggregate them
        network.append(dict(type='retrieve', tensors=tensors, aggregation=aggregation))

        for i in range(layers):
            network.append(dict(type='dense', size=units, activation=activation, dropout=dropout))

        return network

    @staticmethod
    def complex_network(networks: [[dict]], layers=2, units=64, activation='relu', dropout=0.0, aggregation='concat'):
        network = networks
        outputs = []

        # find register (output) layers
        for net in networks:
            layer = net[-1]
            assert layer['type'] == 'register'

            outputs.append(layer['tensor'])

        # aggregate them
        network.append(dict(type='retrieve', tensors=outputs, aggregation=aggregation))

        for i in range(layers):
            network.append(dict(type='dense', size=units, activation=activation, dropout=dropout))

        return network

    @staticmethod
    def policy(network: dict, distributions: str, temperature=0.0):
        return dict(type='parametrized_distributions',
                    distributions=dict(type=distributions),
                    network=network,
                    temperature=temperature)

    @staticmethod
    def saver():
        raise NotImplementedError

    @staticmethod
    def summarizer():
        raise NotImplementedError
