import os
import pygame
import logging
import tensorflow as tf

import carla
import tests

from tensorforce import Agent, Environment
from agents import env_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


# TODO [project-level]: add assertions, docstrings, and annotate functions with types!!
# TODO [project-level]: use os.path.join for directory strings!!
# TODO [project-level]: use logger and different logging levels for debug, warnings, events, etc.!!

if __name__ == '__main__':
    # TODO: to make tensorflow 2.x works comment line 29 and 30 in
    #  '/usr/lib/python3.8/site-packages/tensorflow_core/lite/experimental/microfrontend/python/ops
    #  /audio_microfrontend_op.py'

    # ./CarlaUE4.sh -windowed -ResX=8 -ResY=8 -benchmark -fps=30 --quality-level=Low
    # https://docs.unrealengine.com/en-US/Programming/Basics/CommandLineArguments/index.html

    # Test cases:
    # tests.test_keyboard_agent()
    # tests.ppo_experiment(num_episodes=30, num_timesteps=512, load=True, image_shape=(75, 105, 3))  # 70k steps
    # tests.toy_example()
    # tests.new_env(episodes=3)
    # tests.gym_test(1_000)

    # tests.complete_state(num_episodes=2, num_timesteps=64, image_shape=(75, 105, 3), time_horizon=10, batch_size=1,
    #                      optimization_steps=1)

    # TODO: record experience with different vehicles and weather!
    # tests.collect_experience(num_episodes=34, num_timesteps=512, image_shape=(75, 105, 3))
    # tests.collect_experience(num_episodes=128, num_timesteps=512, image_shape=(75, 105, 3),
    #                          traces_dir='data/traces/pretrain-behaviour')

    TRACES_DIR = 'data/traces/pretrain-ppo3-complete'

    # tests.pretrain_then_train(num_episodes=10, num_timesteps=512, image_shape=(75, 105, 3), time_horizon=10,
    #                           traces_dir='data/traces/pretrain-ppo3-complete',
    #                           # num_iterations=120, save_every=20,
    #                           num_iterations=1, save_every=1,
    #                           # skip_pretraining=True,
    #                           load=True,
    #                           weights_dir='weights/agents/pretrain-ppo4')

    # tests.alternate_training_with_pretraining(train_episodes=8, num_timesteps=512, pretrain_episodes=40,
    #                                           load_agent=True,
    #                                          traces_dir=TRACES_DIR, save_every=20, repeat=1, image_shape=(75, 105, 3),
    #                                           agent_name='alternate', weights_dir='weights/ppo4', record_dir=None)

    # tests.alternate_training_with_pretraining(train_episodes=2, num_timesteps=512, pretrain_episodes=4,
    #                                           load_agent=True,
    #                                     traces_dir=TRACES_DIR, save_every=4, repeat=10 // 2, image_shape=(75, 105, 3),
    #                                           agent_args=dict(learning_rate=1e-4, subsampling_fraction=0.3,
    #                                                           optimization_steps=4),
    #                                           agent_name='alternate_1_2', weights_dir='weights/ppo4', record_dir=None)

    # TODO: enhancements and suggestions
    # 1. take the semantic segmentation image, mark lanes and road in two different (distant) colors, and the rest as
    #    one unique (distant) color to signal them as obstacle (as the same unimportant stuff).
    # 2. reduce the dimensionality of the state space, i.e. reduce the number of features.
    # 3. increase model capacity.
    # 4. lower learning rate on policy (e.g. 3e-5, 1e-5).
    # 5. Embed skills: embedding -> dense ... dense -> out
    # 6. try to add depth information to semantic/camera ones.
    # 7. use "skip trick" but let the agent predict how many times action a should be repeated; a=[control | skill | k]
    # 8. reduce the set of possible spawn points to few random ones (picked before training)
    # 9. embed (road) features
    # 10. edit reward function, use log(speed) or s/10 instead of dividing by it. check collision, action penalty too...
    # 11. bird-eye image observation as in the Berkeley paper..
    # 12. add two more cameras: left view, and right view

    # tests.alternate_training_with_pretraining2(train_episodes=4, num_timesteps=256, pretrain_episodes=8,
    #                                            load_agent=False,
    #                                            traces_dir=TRACES_DIR, save_every=8, repeat=10 // 2,
    #                                            image_shape=(75, 105, 3),
    #                                            agent_args=dict(optimization_steps=5, batch_size=2, entropy=0.1,
    #                                                            radar_shape=(50, 40)),
    #                                            agent_name='alternate', weights_dir='weights/ppo5', record_dir=None)

    # SkipTrickExperiment.run(num_episodes=3, num_timesteps=64//4,
    #                         env_args=dict(image_shape=(75, 105, 3), radar_shape=(50, 40//2), time_horizon=10//2,
    #                                       window_size=(670, 500)),
    #                         agent_args=dict(batch_size=1, optimization_steps=4))

    # tests.test_saver(num_episodes=500)
    # tests.curriculum_learning(batch_size=1, random_seed=31)   # [42, 31]

    # tests.collect_traces2_stage1(num_traces=1, traces_dir='data/traces/stage1', time_horizon=5)

    # seeds 31, 42
    tests.curriculum_learning2(batch_size=1, random_seed=71, time_horizon=5, timesteps=400,
                               weights_dir='weights/curriculum3')

    pygame.quit()
