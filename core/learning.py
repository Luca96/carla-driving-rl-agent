import os
import time
import random

from rl.environments import CARLACollectWrapper
from rl.environments.carla import env_utils as carla_utils
from rl.parameters import *
from rl import utils

from core import CARLAEnv, CARLAgent

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


def define_env(image_shape=(90, 120, 3), render=True, town: Union[None, str] = 'Town01', window_size=(1080, 270),
               debug=False, **kwargs) -> dict:
    return dict(class_=CARLAEnv, debug=debug, window_size=window_size, render=render, town=town,
                image_shape=image_shape, **kwargs)


# -------------------------------------------------------------------------------------------------
# -- Stage
# -------------------------------------------------------------------------------------------------

class Stage:
    """A Curriculum Learning stage"""
    def __init__(self, agent: dict, environment: dict, learning: dict, representation: dict = None,
                 collect: dict = None, imitation: dict = None, name='Stage'):
        assert isinstance(agent, dict)
        assert isinstance(environment, dict)
        assert isinstance(learning, dict)

        # Agent
        self.agent_class = agent.pop('class', agent.pop('class_'))
        self.agent_args = agent
        self.agent = None

        assert isinstance(learning['agent'], dict)

        # Environment
        self.env_class = environment.pop('class', environment.pop('class_'))
        self.env_args = environment
        self.env = None

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

        # Supervised Imitation:
        if isinstance(imitation, dict):
            self.should_imitate = True
            self.imitation_args = imitation
        else:
            self.should_imitate = False

        self.learn_args = learning
        self.name = name

    def init(self):
        self.env = self.env_class(**self.env_args)
        self.agent = self.agent_class(self.env, **self.agent_args)

    def run(self, epochs: int, collect: Union[bool, int] = True, representation=True):
        assert epochs > 0
        self.init()

        if (collect is False) or (not self.should_collect):
            collect = 0
        elif collect is True:
            collect = epochs + 1

        for epoch in range(epochs):
            t0 = time.time()

            # collect -> representation learning -> rl
            if collect > 0:
                self.collect()
                collect -= 1

            if self.should_do_repr_lear and representation:
                self.representation_learning()

            self.reinforcement_learning()
            print(f'[Stage] Epoch {epoch + 1}/{epochs} took {round(time.time() - t0, 3)}s.')

        self.cleanup()

    def run2(self, epochs: int, copy_weights=True, epoch_offset=0):
        assert epochs > 0
        self.init()

        for epoch in range(epochs):
            t0 = time.time()

            if self.should_imitate:
                self.imitation_learning()

            self.reinforcement_learning()
            print(f'[{self.name}] Epoch {epoch + 1}/{epochs} took {round(time.time() - t0, 3)}s.')

            if copy_weights:
                utils.copy_folder(src=self.agent.base_path, dst=f'{self.agent.base_path}-{epoch + epoch_offset}')

        self.cleanup()

    def collect(self):
        wrapper = CARLACollectWrapper(env=self.env, **self.collect_args)
        wrapper.collect(**self.learn_args['collect'])

    def representation_learning(self):
        self.agent.learn_representation(**self.repr_args)

    def imitation_learning(self):
        self.agent.imitation_learning(**self.imitation_args)

    def reinforcement_learning(self):
        self.agent.learn(**self.learn_args['agent'])

    def cleanup(self):
        self.env.close()
        self.env = None
        self.agent = None


# -------------------------------------------------------------------------------------------------
# -- Imitation Learning
# -------------------------------------------------------------------------------------------------

def collect_experience(episodes: int, timesteps: int, threshold=0.0, env_class=CARLAEnv, ignore_traffic_light=True,
                       traces_dir='traces', behaviour='normal', name='collect', **kwargs):
    """
    :param episodes: how much traces to collect.
    :param timesteps: the size in timesteps (number of states/transitions) of a single trace.
    :param threshold: use to discard the traces that yields a total reward that is less than `timesteps * threshold`.
    :param env_class: which CARLAEnvironment to use.
    :param ignore_traffic_light: whether or not the privileged agent should ignore the traffic lights.
    :param traces_dir: where to store the collected experience. Full path is given by `{traces_dir}\\{name}`.
    :param behaviour: the privileged agent's behaviour, must be one of 'cautious', 'normal', 'aggressive'.
    """
    assert behaviour in ['cautious', 'normal', 'aggressive']
    assert 0.0 <= threshold <= 1.0

    wrapper = CARLACollectWrapper(env_class(**kwargs), ignore_traffic_light=ignore_traffic_light, traces_dir=traces_dir,
                                  name=name, behaviour=behaviour)
    wrapper.collect(episodes, timesteps, episode_reward_threshold=threshold)


def imitation_learning(batch_size=64, lr=1e-3, alpha=0.5, beta=0.5, clip=1.0, epochs=1, load=False,
                       name='imitation', polyak=0.99, **kwargs):
    """Performs imitation learning on the already recorded experience traces (given by `name` argument).
        - refer to CARLAgent.imitation_learning(...) for explained arguments.
    """
    env = CARLAEnv(render=True, image_shape=(90, 120, 3), window_size=(720, 180), debug=False, range_controls={})

    agent = CARLAgent(env, batch_size=batch_size, name=name, consider_obs_every=1, load=load,
                      drop_batch_remainder=False)

    agent.imitation_learning(alpha=alpha, beta=beta, clip_grad=clip, epochs=epochs, shuffle_data=True, polyak=polyak,
                             lr=StepDecay(initial_value=lr, decay_steps=100, decay_rate=0.5, min_value=1e-5), **kwargs,
                             traces_dir=os.path.join('traces', name))


# -------------------------------------------------------------------------------------------------
# -- Curriculum (stages definition)
# -------------------------------------------------------------------------------------------------

def stage_s1(episodes: int, timesteps: int, save_every=None, seed=42, **kwargs):
    """Stage-1: origins (n=10) fixed by seed. Town-3, reverse gear disabled, steering within (-0.3, +0.3).
                No dynamic objects."""
    agent_dict = define_agent(
        class_=CARLAgent, **kwargs,
        batch_size=64, name='stage-s1', traces_dir=None, load=True, seed=seed,
        optimization_steps=(1, 1),
        advantage_scale=2.0,
        policy_lr=3e-4, value_lr=3e-4,
        entropy_regularization=0.1, shuffle_batches=False, drop_batch_remainder=False, shuffle=True,
        clip_ratio=0.20, consider_obs_every=1, clip_norm=0.5, update_dynamics=False)

    env_dict = define_env(town=None, debug=True,
                          path=dict(origin=sample_origins(amount=10, seed=seed)),
                          range_controls=dict(steer=(-0.3, 0.3)),
                          info_every=kwargs.get('repeat_action', 1),
                          disable_reverse=True, window_size=(900, 245))

    return Stage(agent=agent_dict, environment=env_dict,
                 learning=dict(agent=dict(episodes=episodes, timesteps=timesteps, render_every=False, close=False,
                                          save_every=save_every)))


def stage_s2(episodes: int, timesteps: int, save_every=None, seed=None, **kwargs):
    """Stage-2: 50 random origins + 50 pedestrians"""
    agent_dict = define_agent(
        class_=CARLAgent, **kwargs,
        batch_size=64, name='stage-s2', traces_dir=None, load=True, seed=seed,
        optimization_steps=(1, 1),
        advantage_scale=2.0,
        policy_lr=3e-4, value_lr=3e-4,
        entropy_regularization=0.1, shuffle_batches=False, drop_batch_remainder=False, shuffle=True,
        clip_ratio=0.20, consider_obs_every=1, clip_norm=0.5, update_dynamics=False)

    env_dict = define_env(town=None, debug=True,
                          path=dict(origin=sample_origins(amount=50, seed=seed)),
                          spawn=dict(vehicles=0, pedestrians=50),
                          range_controls=dict(steer=(-0.3, 0.3)),
                          info_every=kwargs.get('repeat_action', 1),
                          disable_reverse=True, window_size=(900, 245))

    return Stage(agent=agent_dict, environment=env_dict,
                 learning=dict(agent=dict(episodes=episodes, timesteps=timesteps, render_every=False, close=False,
                                          save_every=save_every)))


def stage_s3(episodes: int, timesteps: int, save_every=None, seed=None, **kwargs):
    """Stage-3: random origin with 50 vehicles and 50 pedestrians"""
    agent_dict = define_agent(
        class_=CARLAgent, **kwargs,
        batch_size=64, name='stage-s3', traces_dir=None, load=True, seed=seed,
        optimization_steps=(1, 1),
        advantage_scale=2.0,
        policy_lr=3e-4, value_lr=3e-4, entropy_regularization=0.1,
        shuffle_batches=False, drop_batch_remainder=False, shuffle=True,
        clip_ratio=0.20, consider_obs_every=1, clip_norm=0.5, update_dynamics=False)

    env_dict = define_env(town=None, debug=True,
                          spawn=dict(vehicles=50, pedestrians=50),
                          range_controls=dict(steer=(-0.3, 0.3)),
                          info_every=kwargs.get('repeat_action', 1),
                          disable_reverse=True, window_size=(900, 245))

    return Stage(agent=agent_dict, environment=env_dict,
                 learning=dict(agent=dict(episodes=episodes, timesteps=timesteps, render_every=False, close=False,
                                          save_every=save_every)))


def stage_s4(episodes: int, timesteps: int, town: str, save_every=None, seed=None, **kwargs):
    """Stage-4: new town with regular traffic (50 vehicles and 50 pedestrians)"""
    agent_dict = define_agent(
        class_=CARLAgent, **kwargs,
        batch_size=64, name='stage-s4', traces_dir=None, load=True, seed=seed,
        optimization_steps=(1, 1),
        advantage_scale=2.0,
        policy_lr=3e-4, value_lr=3e-4, entropy_regularization=0.1,
        shuffle_batches=False, drop_batch_remainder=False, shuffle=True,
        clip_ratio=0.20, consider_obs_every=1, clip_norm=0.5, update_dynamics=False)

    env_dict = define_env(town=town, debug=True,
                          spawn=dict(vehicles=50, pedestrians=50),
                          range_controls=dict(steer=(-0.3, 0.3)),
                          info_every=kwargs.get('repeat_action', 1),
                          disable_reverse=True, window_size=(900, 245))

    return Stage(agent=agent_dict, environment=env_dict,
                 learning=dict(agent=dict(episodes=episodes, timesteps=timesteps, render_every=False, close=False,
                                          save_every=save_every)))
