"""Test cases"""

import os
import random
import tensorflow as tf

from rl.environments.carla.environment import ThreeCameraCARLAEnvironment, CARLACollectWrapper

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

def curriculum_learning():
    # Agent
    agent_dict = dict(class_=CARLAgent, policy_lr=1e-3, value_lr=3e-4, optimization_steps=(1, 2), clip_ratio=0.2,
                      gamma=0.99, lambda_=0.95, target_kl=None, entropy_regularization=0.001, drop_batch_reminder=True,
                      skip_data=1, consider_obs_every=4)

    # Environment
    env_dict = dict(class_=ThreeCameraCARLAEnvironment, image_shape=None, path=None, render=True, debug=True,
                    vehicle_filter='vehicle.tesla.model3')

    # Imitation learning
    imitation_dict = dict(class_=CARLAImitationLearning, policy_lr=3e-4, value_lr=3e-4, name='carla-imitation',
                          drop_batch_reminder=True, skip_data=1, consider_obs_every=4)

    # Stage 1: same car, fixed origin and destination. Reverse gear is disabled.
    Stage(agent_dict, env_dict, imitation_dict).run()


def collect_experience():
    pass
