"""Test cases"""

import os
import random
import tensorflow as tf

from rl.environments.carla import env_utils
from rl.parameters import LinearParameter
from rl.environments.carla.environment import ThreeCameraCARLAEnvironment, CARLACollectWrapper

from tensorflow.keras.optimizers import schedules

from agent.agent import CARLAgent, CARLAImitationLearning
from agent.curriculum import Stage


# -------------------------------------------------------------------------------------------------
# -- Curriculum Learning
# -------------------------------------------------------------------------------------------------

# def curriculum_learning_ppo9(batch_size: int, horizon: int, random_seed: int, weights_dir: str, discount=1,
#                              image_shape=(75, 105, 3), time_horizon=5, timesteps=1792, memory=None):
#     if random_seed is not None:
#         tf.compat.v1.random.set_random_seed(random_seed)
#
#     random.seed(42)
#     print(f'random seed = 42, tf.random_seed = {random_seed}')
#
#     world_map = env_utils.get_client(address='localhost', port=2000).get_world().get_map()
#     spawn_point, destination = _get_origin_destination(world_map)
#     print('origin', spawn_point)
#     print('destination', destination)
#
#     cl = CurriculumLearning(agent_spec=dict(callable=Agents.ppo9, batch_size=batch_size, summarizer=Specs.summarizer(),
#                                             optimization_steps=10, entropy=0.0, critic_lr=3e-5, lr=1e-5, clipping=0.25,
#                                             optimizer='adam', noise=0.2, capacity=memory, discount=discount,
#                                             decay=dict(clipping=dict(steps=10_000, type='linear'),
#                                                        noise=dict(steps=200, type='linear'),
#                                                        lr=dict(steps=10_000, type='linear')),
#                                             subsampling_fraction=0.2,
#                                             # recorder={'directory': 'data/traces/ppo8', 'max-traces': 128}
#                                             ),
#
#                             env_spec=dict(callable=MyCARLAEnvironmentNoSkill, max_timesteps=timesteps,
#                                           image_shape=image_shape, window_size=(670, 500), time_horizon=time_horizon),
#
#                             curriculum=[
#                                 # stage 1: fixed car, origin, destination. Reverse gear is disabled.
#                                 dict(environment=dict(vehicle_filter='vehicle.tesla.model3',
#                                                       disable_reverse=True, max_validity=10.0, validity_cap=10.0,
#                                                       path=dict(origin=spawn_point, destination=destination)),
#
#                                      pretrain=dict(traces_dir='data/traces/stage1', num_traces=128, times=0),
#                                      learn_episodes=64, eval_episodes=5, target_reward=10.0, success_rate=0.5,
#                                      repeat=8),
#
#                                 # stage 2: add reverse?
#                                 dict(environment=dict(vehicle_filter='vehicle.tesla.model3',
#                                                       disable_reverse=False, max_validity=10.0, validity_cap=10.0,
#                                                       path=dict(origin=spawn_point, destination=destination)),
#
#                                      # pretrain=dict(traces_dir='data/traces/stage1', num_traces=128, times=0),
#                                      learn_episodes=64, eval_episodes=5, target_reward=10.0, success_rate=0.5,
#                                      repeat=8)
#
#                                 # stage 3: generalize on different vehicles?
#                                 # stage 4: same origin, different destination.
#                                 # stage 5: random (origin, destination)
#                                 # stage 6: add vehicles
#                                 # stage 7: add pedestrians
#                                 # stage 8: generalize on multiple maps
#                                 # stage 9: generalize on multiple weathers?
#                             ],
#                             save=dict(directory=weights_dir, filename='ppo8', frequency=32))
#     # optimizer: adadelta, adamax, adam
#     cl.start()
#
#
# def collect_traces2_stage1(num_traces: int, traces_dir: str, time_horizon=5, timesteps=400, image_shape=(75, 105, 3)):
#     random.seed(42)
#     world_map = env_utils.get_client(address='localhost', port=2000).get_world().get_map()
#     spawn_point, destination = _get_origin_destination(world_map)
#
#     env = CARLACollectTracesNoSkill(max_timesteps=timesteps, vehicle_filter='vehicle.tesla.model3',
#                                     time_horizon=time_horizon, window_size=(670, 500),
#                                     path=dict(origin=spawn_point, destination=destination),
#                                     # path=dict(origin=dict(point=spawn_point, type='route'), destination=destination),
#                                     image_shape=image_shape)
#
#     env.collect(num_traces, traces_dir)


def _get_origin_destination(world_map):
    available_points = world_map.get_spawn_points()
    spawn_point = random.choice(available_points)
    random.shuffle(available_points)
    destination = random.choice(available_points).location
    return spawn_point, destination


# -------------------------------------------------------------------------------------------------

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


if __name__ == '__main__':
    # Stage 1:
    # collect_experience_stage1()
    # curriculum_learning_stage1()

    # Stage x:
    collect_experience_stagex()
    pass
