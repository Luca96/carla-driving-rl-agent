import os
import pygame
import logging
import tensorflow as tf

import tests


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


# TODO [project-level]: add assertions, docstrings, and annotate functions with types!!
# TODO [project-level]: use os.path.join for directory strings!!
# TODO [project-level]: use logger and different logging levels for debug, warnings, etc.!!

if __name__ == '__main__':
    # TODO: to make tensorflow 2.x works comment line 29 and 30 in
    #  '/usr/lib/python3.8/site-packages/tensorflow_core/lite/experimental/microfrontend/python/ops
    #  /audio_microfrontend_op.py'

    # ./CarlaUE4.sh -windowed -ResX=8 -ResY=8 -benchmark -fps=30 --quality-level=Low
    # https://docs.unrealengine.com/en-US/Programming/Basics/CommandLineArguments/index.html

    # Test cases:
    # tests.test_carla_env()
    # tests.test_baseline_env()
    # tests.test_sequence_layer()
    # tests.test_pretrain_env()
    # tests.test_keyboard_agent(version=2)
    # tests.carla_wrapper()
    # tests.ppo_experiment(num_episodes=30, num_timesteps=512, load=True, image_shape=(75, 105, 3))  # 70k steps
    # tests.complete_state(num_episodes=5, num_timesteps=512, load=True, image_shape=(75, 105, 3), time_horizon=10,
    #                      optimization_steps=5)

    pygame.quit()
