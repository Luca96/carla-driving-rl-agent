"""Test cases"""

import os
import random
import tensorflow as tf

from rl.environments.carla import env_utils
from rl.parameters import LinearParameter
from rl.environments.carla.environment import ThreeCameraCARLAEnvironment, CARLACollectWrapper, \
                                              ThreeCameraCARLAEnvironmentDiscrete

from tensorflow.keras.optimizers import schedules

from core.agent import CARLAgent, CARLAImitationLearning
from core.curriculum import Stage


def benchmark_networks(batch_size: int, summary=False, depth=3, **kwargs):
    import time
    from core.networks import shufflenet_v2

    def measure(code, repetitions=10):
        t = 0.0
        for _ in range(repetitions):
            t0 = time.time()
            code()
            t += time.time() - t0
        print(f' - Time passed {round(t / repetitions, 3)}s.')

    shapes = [(105, 420, depth), (90, 360, depth), (75, 300, depth)]
    models = [shufflenet_v2(shape, 1, **kwargs) for shape in shapes]
    datasets = [tf.random.normal((batch_size,) + shape) for shape in shapes]

    if summary:
        for model in models:
            model.summary()
            breakpoint()

    for model, shape, data in zip(models, shapes, datasets):
        print(f'Measuring for batch {(batch_size,) + shape}:')
        measure(code=lambda: model(data))


# -------------------------------------------------------------------------------------------------
# -- Curriculum Learning
# -------------------------------------------------------------------------------------------------

def _get_origin_destination(world_map):
    available_points = world_map.get_spawn_points()
    spawn_point = random.choice(available_points)
    random.shuffle(available_points)
    destination = random.choice(available_points).location
    return spawn_point, destination


def curriculum_learning_stage1():
    """Stage 1: same car, fixed origin and destination. Reverse gear is disabled. ClearNoon weather, Town01 map."""
    random.seed(31)
    world_map = env_utils.get_client(address='localhost', port=2000, timeout=5.0).get_world().get_map()

    spawn_point, destination = _get_origin_destination(world_map)

    # Agent
    agent_dict = dict(class_=CARLAgent,
                      args=dict(policy_lr=schedules.ExponentialDecay(1e-3, decay_steps=5_000, decay_rate=0.9,
                                                                     staircase=True),
                                value_lr=schedules.ExponentialDecay(3e-4, decay_steps=5_000, decay_rate=0.9,
                                                                    staircase=True),
                                noise=LinearParameter(initial=1.0, final=0.001, rate=0.1, steps=1000, restart=True,
                                                      decay_on_restart=0.9),
                                batch_size=50, aug_intensity=0.5, seed=42,
                                optimization_steps=(1, 1), clip_ratio=0.2, gamma=0.99, lambda_=0.95, target_kl=None,
                                entropy_regularization=0.01, recurrent_policy=True, skip_data=1,
                                consider_obs_every=4, load=False, weights_dir='weights', name='stage-1'),
                      learn=dict(episodes=50, render_every=False, timesteps=500, save_every=10))

    # Environment
    env_dict = dict(class_=ThreeCameraCARLAEnvironment,
                    args=dict(image_shape=(105, 140, 1), window_size=(720, 210), render=True, debug=True,
                              path=dict(origin=spawn_point, destination=destination),
                              vehicle_filter='vehicle.tesla.model3', disable_reverse=True))

    # Imitation learning
    imitation_dict = dict(class_=CARLAImitationLearning,
                          args=dict(policy_lr=schedules.ExponentialDecay(1e-3, decay_steps=2_000, decay_rate=0.9,
                                                                         staircase=True),
                                    value_lr=schedules.ExponentialDecay(1e-3, decay_steps=2_000, decay_rate=0.9,
                                                                        staircase=True),
                                    name='stage-1', aug_intensity=0.75,
                                    drop_batch_reminder=True, skip_data=1, consider_obs_every=4),
                          learn=dict(repetitions=30 * 0, shuffle_traces=True, discount=0.99, save_every=10))

    # Stage 1: same car, fixed origin and destination. Reverse gear is disabled.
    Stage(agent_dict, env_dict, imitation_dict).run()


def collect_experience_stage1():
    """Stage 1: same car, fixed origin and destination. Reverse gear is disabled. ClearNoon weather, Town01 map."""
    random.seed(31)
    world_map = env_utils.get_client(address='localhost', port=2000).get_world().get_map()

    spawn_point, destination = _get_origin_destination(world_map)

    CARLACollectWrapper(ThreeCameraCARLAEnvironment(debug=False, window_size=(720, 210), render=True,
                                                    image_shape=(105, 140, 1),
                                                    path=dict(origin=spawn_point, destination=destination)),
                        ignore_traffic_light=True, name='stage-1') \
        .collect(episodes=1, timesteps=500, episode_reward_threshold=15.0 * 480)


def collect_experience_stagex():
    """Stage x: navigation without pedestrians and vehicles."""
    # random.seed(77)
    CARLACollectWrapper(ThreeCameraCARLAEnvironment(debug=False, window_size=(720, 210), render=True,
                                                    image_shape=(105, 140, 1)),
                        ignore_traffic_light=True, name='stage-x') \
        .collect(episodes=15, timesteps=500, episode_reward_threshold=15.0 * 480)  # 50


# -------------------------------------------------------------------------------------------------

def collect_experience_stage1_discrete(episodes=1, bins=8, timesteps=500, image_shape=(105, 140, 3)):
    """Stage 1: same car, one fixed origin and destination. Reverse gear is disabled. ClearNoon weather, Town01 map."""
    random.seed(123)
    world_map = env_utils.get_client(address='localhost', timeout=5.0, port=2000).get_world().get_map()

    spawn_point, destination = _get_origin_destination(world_map)

    env = ThreeCameraCARLAEnvironmentDiscrete(bins=bins, debug=False, window_size=(720, 210), render=True,
                                              image_shape=image_shape, town='Town01',
                                              path=dict(origin=spawn_point, destination=destination))

    CARLACollectWrapper(env, ignore_traffic_light=True, name='stage-1-d') \
        .collect(episodes=episodes, timesteps=timesteps, episode_reward_threshold=15.0 * timesteps)


if __name__ == '__main__':
    # Benchmarks:
    # benchmark_networks(batch_size=64, summary=True)  # 1.80s, 1.45s, 1.20s
    # benchmark_networks(batch_size=64, depth=1)  # 1.64s, 1.32s, 1.09s
    # benchmark_networks(batch_size=64, dilation=(2, 2))  # 2.93s, 2.27s, 1.78s
    # benchmark_networks(batch_size=128, depth=3)  # 2.69s, 2.09s, 2.27s
    # benchmark_networks(batch_size=128, depth=1)  # 2.70s, 2.09s, 1.67s

    # -------------------------------------------
    # Stage 1:
    # collect_experience_stage1()
    # curriculum_learning_stage1()

    # Stage x:
    # collect_experience_stagex()

    # -------------------------------------------
    # Stage 1-d:
    # collect_experience_stage1_discrete()
    pass
