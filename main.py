import os
import carla
import pygame
import logging
import tensorflow as tf

from agents.agents import Agents
from agents.learn.experiments import *
from agents.specifications import Specifications as Specs

from worlds import World
from worlds.controllers import KeyboardController

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


# TODO [project-level]: add assertions!!
# TODO [project-level]: use os.path.join for directory strings!!
# TODO [project-level]: use logger and different logging levels for debug, warnings, etc.!!

def get_client(address='localhost', port=2000, timeout=2.0):
    """Connects to the simulator.
        @:returns a client object if the CARLA simulator accepts the connection.
    """
    client = carla.Client(address, port)
    client.set_timeout(timeout)
    return client


def print_object(obj, message=None, filter=None):
    if message is not None:
        print(message)

    for x in dir(obj):
        if filter is None:
            print(x)
        elif filter in str(type(getattr(obj, x))):
            print(x)


def game_loop(vehicle='vehicle.audi.*', width=800, height=600):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = get_client(timeout=2.0)
        display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)

        world = World(client.get_world(), window_size=(width, height), actor_filter=vehicle)
        controller = KeyboardController()
        # controller = BasicAgentController(vehicle=world.player, destination=world.target_position.location)

        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(30)

            if controller.parse_events(client, world, clock, training=False):
                return
            # control = controller.act()
            # world.apply_control(control)

            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:
        if world and world.recording_enabled:
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


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

    # game_loop()

    num_episodes = 3
    batch_size = 256
    frequency = batch_size
    num_timesteps = batch_size * 3

    # TODO: provide a base CARLA env class for a sync, async, pretrain environments..
    # TODO: do a common CARLA world wrapper for the environments. (or a world_utils module)

    # experiment = CARLABaselineExperiment(window_size=(670, 500), debug=True)
    # experiment.train(agent=Agents.baseline(experiment, batch_size=512),
    #                  num_episodes=10, max_episode_timesteps=512, record_dir=None, weights_dir=None)

    experiment = CARLAExperimentEvo(window_size=(670, 500), debug=True)
    experiment.train(agent=Agents.evolutionary(experiment, 512, update_frequency=128, filters=40, exploration=0.1),
                     num_episodes=10, max_episode_timesteps=512, agent_name='evo', record_dir=None)

    pygame.quit()
