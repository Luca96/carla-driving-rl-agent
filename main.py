import os
import carla
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

def decay(value: float, rate: float, steps: int):
    for _ in range(steps):
        value *= rate

    return value


if __name__ == '__main__':
    # TODO: to make Tensorforce work with tensorflow 2.0.1, comment line 29 and 30 in
    #  '/usr/lib/python3.8/site-packages/tensorflow_core/lite/experimental/microfrontend/python/ops
    #  /audio_microfrontend_op.py'

    # ./CarlaUE4.sh -windowed -ResX=8 -ResY=8 -benchmark -fps=30
    # https://docs.unrealengine.com/en-US/Programming/Basics/CommandLineArguments/index.html

    num_episodes = 3
    batch_size = 256
    frequency = batch_size
    num_timesteps = batch_size * 3

    # experiment = CARLABaselineExperiment(window_size=(670, 500), debug=True)
    # experiment.train(agent=Agents.baseline(experiment, batch_size=512),
    #                  num_episodes=10, max_episode_timesteps=512, record_dir=None, weights_dir=None)

    # experiment = CARLAExperimentEvo(window_size=(670, 500), debug=True)
    # experiment.train(agent=Agents.evolutionary(experiment, 512, update_frequency=256, filters=40, exploration=0.1),
    #                  num_episodes=10, max_episode_timesteps=512, agent_name='evo', record_dir=None, load_agent=True)

    # Pretraining:
    # for i in range(1, 100 + 1):
    #     print(f'Trace-{i}')
    #     experiment = CARLAPretrainExperiment(window_size=(670, 500), debug=True, vehicle_filter='vehicle.tesla.model3')
    #
    #     experiment.train(agent=Agents.dummy.random_walk(experiment, max_episode_timesteps=256, speed=30,
    #                                                     traces_dir='data/traces/pretrain_tesla_batch256'),
    #                      num_episodes=1, max_episode_timesteps=256, record_dir=None, weights_dir=None, load_agent=False)

    # tests.test_carla_env()
    # tests.test_sequence_layer()
    tests.test_keyboard_agent()

    pygame.quit()
