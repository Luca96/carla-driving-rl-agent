import os
import time
import random
import carla

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
        if self.env is None:
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

    def run2(self, epochs: int, copy_weights=True, epoch_offset=0) -> 'Stage':
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
        return self

    def evaluate(self, **kwargs) -> 'Stage':
        self.init()
        self.agent.evaluate(**kwargs)
        return self

    def collect(self):
        wrapper = CARLACollectWrapper(env=self.env, **self.collect_args)
        wrapper.collect(**self.learn_args['collect'])

    # def representation_learning(self):
    #     self.agent.learn_representation(**self.repr_args)

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
                       name='imitation', polyak=0.99, time_horizon=1, **kwargs):
    """Performs imitation learning on the already recorded experience traces (given by `name` argument).
        - refer to CARLAgent.imitation_learning(...) for explained arguments.
    """
    env = CARLAEnv(render=True, time_horizon=time_horizon, image_shape=(90, 120, 3), window_size=(720, 180),
                   debug=False, range_controls={})

    agent = CARLAgent(env, batch_size=batch_size, name=name, consider_obs_every=1, load=load,
                      drop_batch_remainder=False)

    agent.imitation_learning(alpha=alpha, beta=beta, clip_grad=clip, epochs=epochs, shuffle_data=True, polyak=polyak,
                             lr=lr, **kwargs, traces_dir=os.path.join('traces', name))


def explore_traces(traces_dir: str, amount=64, seed=None):
    import tensorflow as tf

    amounts = dict(left=amount, right=amount, center=amount)

    def filter_throttle(s, a, r):
        mask = a[:, 0] >= 0.0

        s = {k: utils.to_float(v)[mask] for k, v in s.items()}

        return s, a[mask], r[tf.concat([mask, [True]], axis=0)]

    def shuffle_trace(s: dict, a, r):
        indices = tf.range(start=0, limit=tf.shape(a)[0], dtype=tf.int32)
        indices = tf.random.shuffle(indices)

        for k, v in s.items():
            s[k] = tf.gather(v, indices)

        a = tf.gather(a, indices)
        r = tf.gather(r, tf.concat([indices, [tf.shape(r)[0] - 1]], axis=0))

        return s, a, r

    def mask_reward(r, mask):
        return r[tf.concat([mask, [True]], axis=0)]

    def filter_steering(s, a, r, t=0.1):
        masks = dict(left=a[:, 1] <= -t,
                     right=a[:, 1] >= t,
                     center=(a[:, 1] > -t) & (a[:, 1] < t))

        filtered_data = []

        for k in ['left', 'center', 'right']:
            mask = masks[k]
            taken = int(min(amounts[k], tf.reduce_sum(tf.cast(mask, tf.int32))))
            amounts[k] -= taken

            filtered_data.append(dict(state={k: v[mask][:taken] for k, v in s.items()},
                                      action=a[mask][:taken],
                                      reward=mask_reward(r, mask)[:taken]))
        return filtered_data

    random.seed(seed)
    data = None

    while sum(map(lambda k_: amounts[k_], amounts)) > 0:
        for j, trace in enumerate(utils.load_traces(traces_dir)):
            print(f'trace-{j}')
            print('amounts:', amounts)
            state, action, reward, _ = utils.unpack_trace(trace)
            state, action, reward = filter_throttle(state, utils.to_float(action), reward)
            state, action, reward = shuffle_trace(state, action, reward)
            f_data = filter_steering(state, action, reward)

            if data is None:
                data = f_data
            else:
                for i, d in enumerate(f_data):
                    data[i]['state'] = utils.concat_dict_tensor(data[i]['state'], d['state'])
                    data[i]['action'] = tf.concat([data[i]['action'], d['action']], axis=0)
                    data[i]['reward'] = tf.concat([data[i]['reward'], d['reward']], axis=0)

            if sum(map(lambda k_: amounts[k_], amounts)) <= 0:
                break

    for i, d in enumerate(data):
        print(i, d['action'].shape)

    d = dict(state=utils.concat_dict_tensor(*list(d['state'] for d in data)),
             action=tf.concat(list(d['action'] for d in data), axis=0),
             reward=tf.concat(list(d['reward'] for d in data), axis=0))

    breakpoint()


# -------------------------------------------------------------------------------------------------
# -- Curriculum (stages definition)
# -------------------------------------------------------------------------------------------------

def stage_s1(episodes: int, timesteps: int, batch_size: int, save_every=None, seed=42, stage_name='stage-s1', **kwargs):
    """Stage-1: origins (n=10) fixed by seed. Town-3, reverse gear disabled
                No dynamic objects."""
    policy_lr = kwargs.pop('policy_lr', 3e-4)
    value_lr = kwargs.pop('value_lr', 3e-4)
    clip_ratio = kwargs.pop('clip_ratio', 0.2)
    entropy = kwargs.pop('entropy_regularization', 0.1)
    dynamics_lr = kwargs.pop('dynamics_lr', 3e-4)

    agent_dict = define_agent(
        class_=CARLAgent, **kwargs,
        batch_size=batch_size, name=stage_name, traces_dir=None, seed=seed,
        advantage_scale=2.0,
        policy_lr=policy_lr,
        value_lr=value_lr,
        dynamics_lr=dynamics_lr,
        entropy_regularization=entropy, shuffle_batches=False, drop_batch_remainder=True, shuffle=True,
        clip_ratio=clip_ratio, consider_obs_every=1, clip_norm=1.0, update_dynamics=True)

    env_dict = define_env(town=None, debug=True,
                          image_shape=(90, 120, 3),
                          path=dict(origin=sample_origins(amount=10, seed=seed)),
                          throttle_as_desired_speed=True,
                          # range_controls=dict(steer=(-0.9, 0.9)),
                          info_every=kwargs.get('repeat_action', 1),
                          disable_reverse=True, window_size=(900, 245))

    return Stage(agent=agent_dict, environment=env_dict,
                 learning=dict(agent=dict(episodes=episodes, timesteps=timesteps, render_every=False, close=False,
                                          save_every=save_every)))


def stage_s2(episodes: int, timesteps: int, batch_size: int, save_every=None, seed=42, stage_name='stage-s2', **kwargs):
    """Stage-2: 50 random origins + 50 pedestrians"""
    policy_lr = kwargs.pop('policy_lr', 3e-4)
    value_lr = kwargs.pop('value_lr', 3e-4)
    clip_ratio = kwargs.pop('clip_ratio', 0.2)
    entropy = kwargs.pop('entropy_regularization', 0.1)
    dynamics_lr = kwargs.pop('dynamics_lr', 3e-4)

    agent_dict = define_agent(
        class_=CARLAgent, **kwargs,
        batch_size=batch_size, name=stage_name, traces_dir=None, load=True, seed=seed,
        advantage_scale=2.0, load_full=True,
        policy_lr=policy_lr,
        value_lr=value_lr,
        dynamics_lr=dynamics_lr,
        entropy_regularization=entropy, shuffle_batches=False, drop_batch_remainder=True, shuffle=True,
        clip_ratio=clip_ratio, consider_obs_every=1, clip_norm=1.0, update_dynamics=True)

    env_dict = define_env(town=None, debug=True, throttle_as_desired_speed=True,
                          image_shape=(90, 120, 3),
                          path=dict(origin=sample_origins(amount=50, seed=seed)),
                          spawn=dict(vehicles=0, pedestrians=50),
                          info_every=kwargs.get('repeat_action', 1),
                          disable_reverse=True, window_size=(900, 245))

    return Stage(agent=agent_dict, environment=env_dict,
                 learning=dict(agent=dict(episodes=episodes, timesteps=timesteps, render_every=False, close=False,
                                          save_every=save_every)))


def stage_s3(episodes: int, timesteps: int, batch_size: int, save_every=None, seed=42, stage_name='stage-s3', **kwargs):
    """Stage-3: random origin with 50 vehicles and 50 pedestrians + random light weather"""
    policy_lr = kwargs.pop('policy_lr', 3e-4)
    value_lr = kwargs.pop('value_lr', 3e-4)
    clip_ratio = kwargs.pop('clip_ratio', 0.2)
    entropy = kwargs.pop('entropy_regularization', 0.1)
    dynamics_lr = kwargs.pop('dynamics_lr', 3e-4)

    agent_dict = define_agent(
        class_=CARLAgent, **kwargs,
        batch_size=batch_size, name=stage_name, traces_dir=None, load=True, seed=seed,
        advantage_scale=2.0, load_full=True,
        policy_lr=policy_lr,
        value_lr=value_lr,
        dynamics_lr=dynamics_lr,
        entropy_regularization=entropy, shuffle_batches=False, drop_batch_remainder=True, shuffle=True,
        clip_ratio=clip_ratio, consider_obs_every=1, clip_norm=1.0, update_dynamics=True)

    light_weathers = [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.ClearSunset,
        carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.SoftRainNoon,
        carla.WeatherParameters.SoftRainSunset,
        carla.WeatherParameters.WetNoon,
        carla.WeatherParameters.WetSunset
    ]

    env_dict = define_env(town=None, debug=True, throttle_as_desired_speed=True,
                          image_shape=(90, 120, 3),
                          random_weathers=light_weathers,
                          spawn=dict(vehicles=50, pedestrians=50),
                          info_every=kwargs.get('repeat_action', 1),
                          disable_reverse=True, window_size=(900, 245))

    return Stage(agent=agent_dict, environment=env_dict,
                 learning=dict(agent=dict(episodes=episodes, timesteps=timesteps, render_every=False, close=False,
                                          save_every=save_every)))


def stage_s4(episodes: int, timesteps: int, batch_size: int, towns=None, save_every=None, seed=42,
             stage_name='stage-s4', **kwargs):
    """Stage-4: town with regular traffic (50 vehicles and 50 pedestrians) + random light weather + data-aug"""
    policy_lr = kwargs.pop('policy_lr', 3e-4)
    value_lr = kwargs.pop('value_lr', 3e-4)
    clip_ratio = kwargs.pop('clip_ratio', 0.2)
    entropy = kwargs.pop('entropy_regularization', 0.1)
    dynamics_lr = kwargs.pop('dynamics_lr', 3e-4)

    agent_dict = define_agent(
        class_=CARLAgent, **kwargs,
        batch_size=batch_size, name=stage_name, traces_dir=None, load=True, seed=seed,
        advantage_scale=2.0, load_full=True,
        policy_lr=policy_lr,
        value_lr=value_lr,
        dynamics_lr=dynamics_lr,
        entropy_regularization=entropy, shuffle_batches=False, drop_batch_remainder=True, shuffle=True,
        clip_ratio=clip_ratio, consider_obs_every=1, clip_norm=1.0, update_dynamics=True)

    light_weathers = [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.ClearSunset,
        carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.SoftRainNoon,
        carla.WeatherParameters.SoftRainSunset,
        carla.WeatherParameters.WetNoon,
        carla.WeatherParameters.WetSunset
    ]

    env_dict = define_env(town=None, debug=True, throttle_as_desired_speed=True,
                          image_shape=(90, 120, 3), random_towns=towns,
                          random_weathers=light_weathers,
                          spawn=dict(vehicles=50, pedestrians=50),
                          info_every=kwargs.get('repeat_action', 1),
                          disable_reverse=True, window_size=(900, 245))

    return Stage(agent=agent_dict, environment=env_dict,
                 learning=dict(agent=dict(episodes=episodes, timesteps=timesteps, render_every=False, close=False,
                                          save_every=save_every)))


def stage_s5(episodes: int, timesteps: int, batch_size: int, town: str, save_every=None, seed=42, stage_name='stage-s5',
             weather=None, traffic='dense', **kwargs):
    """Stage-5: town with dense traffic (100 vehicles and 200 pedestrians) + random light weather + data-aug"""
    policy_lr = kwargs.pop('policy_lr', 3e-4)
    value_lr = kwargs.pop('value_lr', 3e-4)
    clip_ratio = kwargs.pop('clip_ratio', 0.2)
    entropy = kwargs.pop('entropy_regularization', 0.1)
    dynamics_lr = kwargs.pop('dynamics_lr', 3e-4)

    agent_dict = define_agent(
        class_=CARLAgent, **kwargs,
        batch_size=batch_size, name=stage_name, traces_dir=None, load=True,
        advantage_scale=2.0, load_full=True, seed=seed,
        policy_lr=policy_lr,
        value_lr=value_lr,
        dynamics_lr=dynamics_lr,
        entropy_regularization=entropy, shuffle_batches=False, drop_batch_remainder=True, shuffle=True,
        clip_ratio=clip_ratio, consider_obs_every=1, clip_norm=1.0, update_dynamics=True)

    light_weathers = [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.ClearSunset,
        carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.SoftRainNoon,
        carla.WeatherParameters.SoftRainSunset,
        carla.WeatherParameters.WetNoon,
        carla.WeatherParameters.WetSunset
    ]

    if weather is None:
        weather = light_weathers

    traffic_spec = dict(no=None,
                        regular=dict(vehicles=100, pedestrians=200),
                        dense=dict(vehicles=100, pedestrians=200))

    env_dict = define_env(town=town, debug=True, throttle_as_desired_speed=True,
                          image_shape=(90, 120, 3),
                          random_weathers=weather,
                          spawn=traffic_spec[traffic],
                          info_every=kwargs.get('repeat_action', 1),
                          disable_reverse=True, window_size=(900, 245))

    return Stage(agent=agent_dict, environment=env_dict,
                 learning=dict(agent=dict(episodes=episodes, timesteps=timesteps, render_every=False, close=False,
                                          save_every=save_every)))


# -------------------------------------------------------------------------------------------------
# -- EVALUATION
# -------------------------------------------------------------------------------------------------

def evaluate(mode: str, town: str, seeds: list, traffic: str, steps=512, trials=50, weights='stage-s5'):
    def make_stage(weather):
        return stage_s5(episodes=1, timesteps=steps, batch_size=1, town=town, seed_regularization=True,
                        stage_name=weights,
                        weather=weather, aug_intensity=0.0, repeat_action=1, traffic=traffic, log_mode=None)

    if mode == 'train':
        weather = None
    else:
        test_weather = [
            carla.WeatherParameters.CloudySunset,
            carla.WeatherParameters.HardRainNoon,
            carla.WeatherParameters.HardRainSunset,
            carla.WeatherParameters.MidRainSunset,
            carla.WeatherParameters.MidRainyNoon,
            carla.WeatherParameters.WetCloudyNoon,
            carla.WeatherParameters.WetCloudySunset
        ]

        weather = test_weather

    stage = make_stage(weather)

    for i, seed in enumerate(seeds):
        stage.evaluate(name=f'{weights}-{mode}-{steps}-{trials}-{town}-{traffic}-{seed}', timesteps=steps,
                       trials=trials, town=None, seeds='sample', initial_seed=seed, close=i + 1 == len(seeds))
