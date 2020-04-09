"""A list of classes that wraps specifications dict for the ease of defining TensorforceAgents agents"""
from tensorforce import Agent

from agents.learn import SynchronousCARLAEnvironment


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
    def convolutional(inputs: [str] = None, output: str = None, initial_filters=32, kernel=(3, 3), pool='max',
                      activation='relu', stride=1, dropout=0.0, layers=2):
        network = []

        if inputs is not None:
            if isinstance(inputs, list) and len(inputs) > 0:
                network.append(dict(type='retrieve', tensors=inputs))
            elif isinstance(inputs, str):
                network.append(dict(type='retrieve', tensors=[inputs]))

        for i in range(1, layers + 1):
            network.append(dict(type='conv2d', size=initial_filters * i, window=kernel, stride=stride,
                                activation=activation, dropout=dropout))

            # TODO: add batch normalization??

            if pool:
                network.append(dict(type='pool2d', reduction=pool))

        network.append(dict(type='pooling', reduction='mean'))

        if output:
            network.append(dict(type='register', tensor=output))

        return network

    @staticmethod
    def dense(inputs: [str] = None, output: str = None, units=64, layers=2, activation='relu', dropout=0.0):
        network = []

        if inputs is not None:
            if isinstance(inputs, list) and len(inputs) > 0:
                network.append(dict(type='retrieve', tensors=inputs))
            elif isinstance(inputs, str):
                network.append(dict(type='retrieve', tensors=[inputs]))

        for i in range(layers):
            network.append(dict(type='dense', size=units, activation=activation, dropout=dropout))

        if output:
            network.append(dict(type='register', tensor=output))

        return network

    @staticmethod
    def complex(networks: [[dict]], layers=2, units=64, activation='relu', dropout=0.0, aggregation='concat',
                rnn: dict = None):
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

        if rnn and rnn.get('length', 0) > 0:
            network.append(dict(type='internal_rnn', cell=rnn.get('cell', 'lstm'), size=rnn.get('units, 128'),
                                length=rnn.get('length'), bias=True, activation=rnn.get('activation', 'none'),
                                dropout=dropout))
        return network


class Agents:
    pass


class Specifications:
    """Explicits TensorForce's specifications as dicts"""
    objectives = Objectives
    optimizers = Optimizers
    networks = Networks
    agents = Agents

    # Short names:
    obj = objectives
    opt = optimizers
    net = networks

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
    def policy(network: dict, distributions: str = None, temperature=0.0, infer_states_value=False):
        return dict(type='parametrized_distributions',
                    infer_states_value=infer_states_value,
                    distributions=dict(type=distributions) if isinstance(distributions, str) else None,
                    network=network,
                    temperature=temperature)

    @staticmethod
    def agent_network(conv: dict, rnn: dict = None, final: dict = None, dropout=0.2):
        # TODO: image stack 4-images (i.e. concat depth)?? or stack last-4 states and actions?
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
    def agent_network_v0(dropout=0.2):
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
    def saver():
        raise NotImplementedError

    @staticmethod
    def summarizer(directory='data/summaries', labels=None, frequency=100):
        # ['graph', 'entropy', 'kl-divergence', 'losses', 'rewards'],
        return dict(directory=directory,
                    labels=labels or ['entropy', 'action-entropies', 'gaussian', 'exploration', 'beta', 'kl-divergences', 'losses', 'rewards'],
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
    def carla_agent(environment: SynchronousCARLAEnvironment, max_episode_timesteps: int, policy: dict, critic: dict,
                    discount=1.0, horizon=100, batch_size=256, update_frequency=64, **kwargs):
        return Agent.create(agent='tensorforce',
                            environment=environment,
                            max_episode_timesteps=max_episode_timesteps,

                            update=Specifications.update(unit='timesteps', batch_size=batch_size,
                                                         frequency=update_frequency,
                                                         start=batch_size),
                            # Policy
                            policy=Specifications.policy(network=policy['network'],
                                                         temperature=policy.get('temperature', 1.0),
                                                         infer_states_value=True),
                            memory=dict(type='recent'),
                            optimizer=policy.get('optimizer', dict(type='adam', learning_rate=3e-4)),
                            objective=Objectives.policy_gradient(clipping_value=0.2, early_reduce=True),

                            # Critic
                            baseline_policy=Specifications.policy(distributions='gaussian',
                                                                  network=critic['network'],
                                                                  temperature=critic.get('temperature', 1.0)),
                            baseline_optimizer=critic.get('optimizer', dict(type='adam', learning_rate=3e-4)),
                            baseline_objective=Objectives.value(value='state', huber_loss=0.1, early_reduce=True),

                            # Reward
                            reward_estimation=dict(discount=discount,
                                                   horizon=horizon,
                                                   estimate_horizon='early',
                                                   estimate_advantage=True),
                            **kwargs)

    # @staticmethod
    # def carla_agent_v1(environment: SynchronousCARLAEnvironment, max_episode_timesteps: int, **kwargs):
    #     return Specifications.carla_agent(environment,
    #                                       max_episode_timesteps,
    #                                       )

    @staticmethod
    def agent_v1(batch_size=256, update_frequency=256, decay_steps=768, filters=36, decay=0.995, lr=0.1,
                 units=(256, 128), layers=(2, 2), temperature=(0.9, 0.7), exploration=0.0):
        ExpDecay = Specifications.exp_decay
        policy_net = Specifications.agent_network(conv=dict(stride=1, pooling='max', filters=filters),
                                                  final=dict(layers=layers[0], units=units[0], activation='leaky-relu'))

        decay_lr = ExpDecay(steps=decay_steps, unit='updates', initial_value=lr, rate=decay)

        critic_net = Specifications.agent_network(conv=dict(stride=1, pooling='max', filters=filters),
                                                  final=dict(layers=layers[1], units=units[1]))

        return dict(policy=dict(network=policy_net,
                                optimizer=dict(type='evolutionary', num_samples=6, learning_rate=decay_lr),
                                temperature=temperature[0]),

                    batch_size=batch_size,
                    update_frequency=update_frequency,

                    critic=dict(network=critic_net,
                                optimizer=dict(type='adam', learning_rate=3e-3),
                                temperature=temperature[1]),

                    discount=1.0,
                    horizon=100,

                    preprocessing=dict(image=[dict(type='image', width=140, height=105, grayscale=True),
                                              dict(type='exponential_normalization')]),

                    summarizer=Specifications.summarizer(frequency=update_frequency),

                    entropy_regularization=ExpDecay(steps=decay_steps, unit='updates', initial_value=lr, rate=decay),
                    exploration=exploration)

    @staticmethod
    def agent_v2(batch_size=256, update_frequency=256, decay_steps=768, filters=36, decay=0.995, lr=0.1,
                 units=(256, 128), layers=(2, 2), temperature=(0.9, 0.7)):
        pass

    @staticmethod
    def agent_v3():
        # TODO: augment agent_v1 with an RNN
        # TODO: also stack 4 input (agent_v3?)
        # TODO: use separable-convolutions
        # TODO: reduce input size of image observation, e.g. 84x84, 105x75
        # TODO: use control instead of previous actions?
        # TODO: architecture: CNN -> concat(features) -> DNN + RNN -> actions

        # (160, 120) -> ~1.5, (140, 105) -> ~2, (100, 75) -> 4
        # preprocessing=dict(image=[dict(type='image', width=140, height=105, grayscale=True),  # 100, 75
        #                           dict(type='deltafier'),
        #                           dict(type='sequence', length=4)],
        #                    vehicle_features=[dict(type='deltafier'),
        #                                      dict(type='sequence', length=4)],
        #                    road_features=[dict(type='deltafier'),
        #                                   dict(type='sequence', length=4)],
        #                    previous_actions=[dict(type='deltafier'),
        #                                      dict(type='sequence', length=4)]),
        pass
