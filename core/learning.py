import random

from rl.environments import ThreeCameraCARLAEnvironmentDiscrete, CARLACollectWrapper, CARLABaseEnvironment
from rl.environments.carla import env_utils as carla_utils
from rl.parameters import schedules

from core.carla_agent import CARLAgent

from typing import Union


# -------------------------------------------------------------------------------------------------
# -- Utils
# -------------------------------------------------------------------------------------------------

def sample_origins(amount=1, seed=None):
    assert amount > 0

    if isinstance(seed, int):
        random.seed(seed)

    client = carla_utils.get_client(address='localhost', port=2000, timeout=10.0)
    world_map = client.get_world().get_map()
    available_points = world_map.get_spawn_points()

    if amount > 1:
        random.shuffle(available_points)
        return available_points[:amount]

    return random.choice(available_points)


def sample_destinations(amount=1, seed=None):
    assert amount > 0

    if isinstance(seed, int):
        random.seed(seed)

    client = carla_utils.get_client(address='localhost', port=2000, timeout=10.0)
    world_map = client.get_world().get_map()
    available_points = world_map.get_spawn_points()

    if amount > 1:
        random.shuffle(available_points)
        return list(map(lambda d: d.location, available_points[:amount]))

    return random.choice(available_points).location


def define_agent(class_=CARLAgent, batch_size=128, consider_obs_every=4, load=False, **kwargs) -> dict:
    return dict(class_=class_, batch_size=batch_size, consider_obs_every=consider_obs_every, skip_data=1, load=load,
                **kwargs)


def define_env(class_=ThreeCameraCARLAEnvironmentDiscrete, bins=6, image_shape=(90, 120, 3), render=True, town='Town01',
               window_size=(720, 180), debug=False, **kwargs) -> dict:
    return dict(class_=class_, bins=bins, debug=debug, window_size=window_size, render=render, town=town,
                image_shape=image_shape, **kwargs)


def get_exp_lr(episodes: int, timesteps: int, rate=0.999, policy=1e-3, value=3e-4, dynamics=1e-3) -> dict:
    d = dict()

    for key, lr in [('policy_lr', policy), ('value_lr', value), ('dynamics_lr', dynamics)]:
        d[key] = schedules.ExponentialSchedule(lr, decay_steps=episodes * timesteps, decay_rate=rate, staircase=False)

    return d


# -------------------------------------------------------------------------------------------------
# -- Stage
# -------------------------------------------------------------------------------------------------

class Stage:
    def __init__(self, agent: dict, environment: dict, learning: dict, representation: dict = None,
                 collect: dict = None):
        assert isinstance(agent, dict)
        assert isinstance(environment, dict)
        assert isinstance(learning, dict)

        # Agent
        # self.agent_class = agent.get('class', agent['class_'])
        # self.agent_args = agent.get('args', {})
        self.agent_class = agent.pop('class', agent.pop('class_'))
        self.agent_args = agent
        self.agent = None

        # assert isinstance(self.agent_class, CARLAgent)
        # assert isinstance(self.agent_args, dict)
        assert isinstance(learning['agent'], dict)

        # Environment
        # self.env_class = environment.get('class', environment['class_'])
        # self.env_args = environment.get('args', {})
        self.env_class = environment.pop('class', environment.pop('class_'))
        self.env_args = environment
        self.env = None

        # assert isinstance(self.env_class, CARLABaseEnvironment)
        # assert isinstance(self.env_args, dict)

        # Representation
        if isinstance(representation, dict):
            self.should_do_repr_lear = True
            self.repr_args = representation
        else:
            self.should_do_repr_lear = False

        # Collect
        if isinstance(collect, dict):
            self.should_collect = True
            self.collect_args = collect

            assert isinstance(learning['collect'], dict)
        else:
            self.should_collect = False

        self.learn_args = learning

    def init(self):
        self.env = self.env_class(**self.env_args)
        self.agent = self.agent_class(self.env, **self.agent_args)

    def run(self, repeat: int, collect: Union[bool, int] = True, representation=True):
        assert repeat > 0
        self.init()

        if (collect is False) or (not self.should_collect):
            collect = 0
        elif collect is True:
            collect = repeat + 1

        for _ in range(repeat):
            # collect -> representation learning -> rl
            if collect > 0:
                self.collect()
                collect -= 1

            if self.should_do_repr_lear and representation:
                self.representation_learning()

            self.reinforcement_learning()

        self.cleanup()

    def collect(self):
        wrapper = CARLACollectWrapper(env=self.env, **self.collect_args)
        wrapper.collect(**self.learn_args['collect'])

    def representation_learning(self):
        self.agent.learn_representation(**self.repr_args)

    def reinforcement_learning(self):
        self.agent.learn(**self.learn_args['agent'])

    def cleanup(self):
        self.env = None
        self.agent = None


# -------------------------------------------------------------------------------------------------
# -- Stage definitions
# -------------------------------------------------------------------------------------------------

def stage_1(episodes: int, timesteps: int, num_destinations=5, seed=None):
    """Stage-1: same origin, [n] different destinations"""
    # TODO: buggy if the client starts with the wrong city!!!
    # destination = dict(points=sample_destinations(num_destinations, seed), type='sequential')

    agent_dict = define_agent(batch_size=timesteps // 4, name='stage-1', traces_dir='traces',
                              optimization_steps=(1, 1), **get_exp_lr(episodes, timesteps),
                              entropy_regularization=0.01, clip_ratio=0.1, tau=0.1, context_size=64,
                              clip_norm=(1.0, 1.0, 1.0))

    env_dict = define_env(path=dict(origin=sample_origins(seed=seed)), town=None, disable_reverse=True)

    return Stage(agent=agent_dict, environment=env_dict,
                 collect=dict(ignore_traffic_light=True, name='stage-1', behaviour='normal'),
                 representation=dict(num_traces=episodes, shuffle_traces=True, batch_size=timesteps // 2,
                                     save_every='end'),
                 learning=dict(agent=dict(episodes=episodes, timesteps=timesteps, render_every=False, close=False,
                                          save_every='end'),
                               collect=dict(episodes=num_destinations, timesteps=timesteps)))
