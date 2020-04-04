import os
import carla
import pygame
import logging
import tensorflow as tf

from tqdm import tqdm
from tensorforce import Runner, Environment, Agent

from agents.pretrain.human import HumanDriverAgent
from agents.learn import *
from agents.specifications import Specifications as Specs

from worlds import World
from worlds.controllers import BasicAgentController, KeyboardController

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


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


def exp_decay(steps, rate, unit='timesteps', initial_value=1.0, increasing=False):
    return dict(type='decaying',
                decay='exponential',
                unit=unit,
                initial_value=initial_value,
                increasing=increasing,
                decay_steps=steps,
                decay_rate=rate)


def test_new_env(agent_builder, timesteps, episodes, vehicle='vehicle.tesla.model3', width=800, height=600,
                 image_shape=(150, 200, 3), **kwargs):
    pygame.init()
    pygame.font.init()

    try:
        environment = CarlaEnvironment(client=get_client(),
                                       image_shape=image_shape,
                                       actor_filter=vehicle,
                                       max_fps=20,
                                       window_size=(800, 600))
        print('== ENV ==')

        # controller = KeyboardController(environment.world)

        agent = agent_builder(environment, max_episode_timesteps=timesteps, **kwargs)
        print('== AGENT ==')

        runner = Runner(agent, environment, max_episode_timesteps=timesteps, num_parallel=None)
        runner.run(num_episodes=episodes)

        agent.save(directory='weights/agents', filename='carla-agent')
        print('agent saved.')

        runner.run(num_episodes=1, save_best_agent='weights/agents', evaluation=True)
        runner.close()

        # for episode in range(episodes):
        #     print('Epoch ' + str(episode))
        #     states = environment.reset()
        #     terminal = False
        #     total_reward = 0.0
        #
        #     for i in range(timesteps):
        #         actions = agent.act(states)
        #         states, terminal, reward = environment.execute(actions)
        #         total_reward += reward
        #         #
        #         # print(f'[{epoch}::{i}::{int(total_reward)}] reward: {round(reward, 2)}, actions: {actions}')
        #         # # print('reward:', reward)
        #         # # print('actions:', actions)
        #         agent.observe(reward, terminal)
        #
        #         if terminal:
        #             break
        #
        #         # Just to have some manual control while learning, can this be used to aid the agents, as a sort of
        #         # supervisory signal? And what about adding a sort of 'manual penalty' to the reward function?
        #         if controller.parse_events(client, world, environment.clock, training=True):
        #             return
        #
        #     # agent.save(directory='weights', filename=f'{agent_type}-car-agent', format='tensorflow', append='episodes')
        #     # print('agent saved.')
        #     # agent.reset()
        #
        # environment.close()

    finally:
        pygame.quit()


def test_sync_event(carla_env, timesteps, episodes, load_agent=False, agent_name='carla-agent',
                    vehicle='vehicle.tesla.model3', width=800, height=600, image_shape=(150, 200, 3), **kwargs):
    pygame.init()
    pygame.font.init()
    sync_env = None

    try:
        sync_env = carla_env(image_shape=image_shape,
                             window_size=(width, height),
                             timeout=5.0,
                             debug=True,
                             fps=15,
                             vehicle_filter=vehicle)
        print('== ENV ==')

        agent = Specs.carla_agent(environment=sync_env, max_episode_timesteps=timesteps, **kwargs)
        print('== AGENT ==')

        if load_agent:
            agent.load(directory='weights/agents', filename=agent_name, environment=sync_env)
            print('Agent loaded.')

        for episode in range(episodes):
            states = sync_env.reset()
            total_reward = 0.0

            with sync_env.synchronous_context:
                # for i in tqdm(range(timesteps), desc=f'E-{episode} [{total_reward}]'):
                for i in range(timesteps):
                    actions = agent.act(states)
                    states, terminal, reward = sync_env.execute(actions)
                    total_reward += reward

                    terminal = terminal or i == timesteps - 1  # hack for recorder

                    # # print('reward:', reward)
                    update_performed = agent.observe(reward, terminal)

                    if update_performed:
                        print(f'{i}/{timesteps} -> update performed.')

                    if terminal:
                        print('Episode ended.\n')
                        break

            print(f'E-{episode} total_reward: {round(total_reward, 2)}')

            agent.save(directory='weights/agents', filename=agent_name)
            print('Agent saved.')

    finally:
        sync_env.close()
        pygame.quit()


def test_pretrain(episodes, timesteps, **kwargs):
    pygame.init()
    pygame.font.init()

    try:
        environment = CarlaEnvironment(client=get_client(),
                                       image_shape=(200, 150, 3),
                                       actor_filter='vehicle.tesla.model3',
                                       max_fps=20 + 10,
                                       window_size=(800, 600))
        print('== ENV ==')

        agent = HumanDriverAgent(environment=environment,
                                 max_episode_timesteps=timesteps,
                                 **kwargs)

        environment.on_reset_event(callback=agent.init_controller)
        print('== AGENT ==')

        runner = Runner(agent, environment, max_episode_timesteps=timesteps)
        runner.run(num_episodes=episodes)
        runner.close()

    finally:
        pygame.quit()


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
    # TODO: do a common CARLA world wrapper for the environments...

    test_sync_event(CARLAExperiment3, num_timesteps, num_episodes * 20,
                    width=670, height=500,
                    load_agent=False, agent_name='carla-agent-evo-f40',

                    # Agent arguments:
                    policy=dict(network=Specs.agent_network(conv=dict(stride=1, pooling='max', filters=36),
                                                            final=dict(layers=2, units=256, activation='leaky-relu')),
                                optimizer=dict(type='evolutionary',
                                               num_samples=5 + 1,
                                               learning_rate=exp_decay(steps=num_timesteps, rate=0.995)),
                                # temperature=exp_decay(steps=num_timesteps, increasing=False, rate=0.995)),
                                temperature=0.90),

                    batch_size=batch_size,
                    update_frequency=frequency,

                    critic=dict(network=Specs.agent_network(conv=dict(stride=1, pooling='max', filters=36),
                                                            final=dict(layers=2, units=160)),
                                optimizer=dict(type='adam', learning_rate=3e-3),
                                # optimizer=dict(type='adam', learning_rate=exp_decay(steps=num_timesteps, rate=0.995)),
                                temperature=0.70),

                    discount=1.0,
                    horizon=100,

                    # (160, 120) -> ~1.5, (140, 105) -> ~2, (100, 75) -> 4
                    # preprocessing=dict(image=dict(type='image', width=140, height=105, grayscale=True)),

                    # preprocessing=dict(image=[dict(type='image', width=140, height=105, grayscale=True),  # 100, 75
                    #                           dict(type='deltafier'),
                    #                           dict(type='sequence', length=4)],
                    #                    vehicle_features=[dict(type='deltafier'),
                    #                                      dict(type='sequence', length=4)],
                    #                    road_features=[dict(type='deltafier'),
                    #                                   dict(type='sequence', length=4)],
                    #                    previous_actions=[dict(type='deltafier'),
                    #                                      dict(type='sequence', length=4)]),

                    preprocessing=dict(image=[dict(type='image', width=140, height=105, grayscale=True),
                                              dict(type='exponential_normalization')],
                                       # vehicle_features=dict(type='exponential_normalization'),
                                       # road_features=dict(type='exponential_normalization'),
                                       # previous_actions=dict(type='exponential_normalization')
                                       ),
                    # TODO: normalize/deltafier rewards?

                    # recorder=dict(directory='data/traces', frequency=1),

                    summarizer=Specs.summarizer(frequency=frequency),

                    entropy_regularization=exp_decay(steps=num_timesteps, rate=0.995, increasing=False),
                    exploration=0.0)
