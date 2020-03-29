import os
import carla
import pygame
import logging
import numpy as np
import tensorflow as tf
import tensorforce
from tensorforce.agents import TensorforceAgent

from tqdm import tqdm
from tensorforce import Runner, Environment, Agent

from agents import AgentConfigs
from agents.pretrain.human import HumanDriverAgent
from agents.learn import CarlaEnvironment, SynchronousCARLAEnvironment
from agents.specifications import Specifications as specs

from worlds import World
from worlds.controllers import BasicAgentController, KeyboardController
from worlds.debug.graphics import HUD

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


def exp_decay(steps, rate, unit='timesteps', initial_value=1.0):
    return dict(type='decaying',
                decay='exponential',
                unit=unit,
                initial_value=initial_value,
                decay_steps=steps,
                decay_rate=rate)


def tensorboard_summarizer(directory='data/summaries', frequency=100):
    return dict(directory=directory,
                labels=['graph', 'entropy', 'kl-divergence', 'losses', 'rewards'],  # or 'all'
                frequency=frequency)


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


def test_sync_event(agent_builder, timesteps, episodes, load_agent=False, agent_name='carla-agent',
                    vehicle='vehicle.tesla.model3', width=800, height=600, image_shape=(150, 200, 3), **kwargs):
    pygame.init()
    pygame.font.init()
    sync_env = None
    runner = None

    try:
        sync_env = SynchronousCARLAEnvironment(image_shape=image_shape,
                                               window_size=(width, height),
                                               timeout=5.0,
                                               debug=True,
                                               fps=15,
                                               vehicle_filter=vehicle)
        print('== ENV ==')

        agent = agent_builder(sync_env, max_episode_timesteps=timesteps, **kwargs)
        print('== AGENT ==')

        if load_agent:
            agent.load(directory='weights/agents', filename=agent_name, environment=sync_env)
            print('Agent loaded.')

        if True:
            for episode in range(episodes):
                states = sync_env.reset()
                terminal = False
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
        else:
            # runner hack:
            sync_env.reset()

            with sync_env.synchronous_context:
                runner = Runner(agent, sync_env, max_episode_timesteps=timesteps)
                runner.run(num_episodes=episodes)

                # runner = Runner(agent, environment=sync_env, num_parallel=2)
                # runner.run(num_episodes=episodes, batch_agent_calls=True)

                # agent.save(directory='weights/agents', filename='carla-agent')
                # print('agent saved.')

                # runner.run(num_episodes=1, save_best_agent='weights/agents', evaluation=True)

    finally:
        if runner:
            runner.close()
        else:
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
    num_timesteps = 256 * 1
    frequency = 64
    batch_size = 256

    # TODO: subclass TensorforceAget -> CARLAAgent(Agent)
    # TODO: provide a base CARLA env class for a sync, async, pretrain environments..
    # TODO: do a common CARLA world wrapper for the envornoments...

    # TODO: make everything on-policy: use recent memory, add internal-RNN, ...

    agent_args = dict(policy=specs.policy(network=specs.agent_network(),
                                          infer_states_value=False,
                                          temperature=0.95),

                      optimizer=dict(type='adam', learning_rate=3e-4),

                      objective=specs.obj.policy_gradient(clipping_value=0.2, early_reduce=True),

                      # update=specs.update(unit='timesteps', batch_size=0, frequency='never'),
                      update=specs.update(unit='timesteps', batch_size=batch_size, frequency=frequency),

                      memory=dict(type='recent'),

                      # Baseline (critic)
                      baseline_policy=specs.policy(distributions='gaussian',
                                                   network=specs.agent_network(),
                                                   infer_states_value=False,
                                                   temperature=0.99),  # TODO: could less stochasticity help estimate V?
                      baseline_optimizer=dict(type='adam', learning_rate=0.0003),
                      baseline_objective=specs.obj.value(value='state', huber_loss=0.1, early_reduce=True),

                      reward_estimation=dict(discount=1.0,
                                             horizon=100,
                                             estimate_horizon='early',
                                             estimate_advantage=True),

                      # preprocessing=dict(previous_actions=dict(type='deltafier'),
                      #                    reward=dict(type='deltafier')),

                      # TODO: se si riduce ulteriormente la taglia dell'immagine, rimuovere qualche conv dal modello
                      # (160, 120) -> ~1.5, (140, 105) -> ~2, (100, 75) -> 4
                      preprocessing=dict(image=dict(type='image', width=140, height=105, grayscale=True)),

                      # preprocessing=dict(image=[dict(type='image', width=140, height=105, grayscale=True),  # 100, 75
                      #                           dict(type='deltafier'),
                      #                           dict(type='sequence', length=4)],
                      #                    vehicle_features=[dict(type='deltafier'),
                      #                                      dict(type='sequence', length=4)],
                      #                    road_features=[dict(type='deltafier'),
                      #                                   dict(type='sequence', length=4)],
                      #                    previous_actions=[dict(type='deltafier'),
                      #                                      dict(type='sequence', length=4)]),

                      # recorder=dict(directory='data/traces', frequency=1),

                      exploration=0.05,  # exp_decay(steps=50, rate=0.5),
                      entropy_regularization=0.1)

    # test_new_env(AgentConfigs.tensorforce2, num_timesteps, num_episodes, **agent_args)
    test_sync_event(AgentConfigs.tensorforce2, num_timesteps, num_episodes * 10 // 10,
                    load_agent=False, agent_name='carla-agent-seq4',
                    **agent_args)

    # test_pretrain(episodes=num_episodes,
    #               timesteps=num_timesteps,
    #
    #               policy=dict(type='parametrized_distributions',
    #                           network=light_net,
    #                           temperature=0.99),
    #
    #               optimizer='adam',
    #               objective=specs.obj.policy_gradient(clipping_value=0.2),
    #
    #               update=specs.update(unit='timesteps', batch_size=batch_size, frequency='never'),
    #
    #               reward_estimation=dict(horizon=1, discount=1.0),
    #
    #               # preprocessing=dict(type='deltafier'),
    #
    #               recorder=dict(directory='data/traces'))

    # ---------------------------------------------------------------------------------------------

# Find actors:
# actor_list = worlds.get_actors()
# # Find an actor by id.
# actor = actor_list.find(id)
# # Print the location of all the speed limit signs in the worlds.
# for speed_sign in actor_list.filter('traffic.speed_limit.*'):
#     print(speed_sign.get_location())

# Each vehicle has a bounding-box:
# box = vehicle.bounding_box
# print(box.location)  # Location relative to the vehicle.
# print(box.extent)  # XYZ half-box extents in meters.

# https://carla.readthedocs.io/en/latest/core_map/
# Reload the same map, or load another map:
# client.reload_world()
# client.load_world('Town0x')
# print(client.get_available_maps())

# Nearest waypoint on the center of a Driving or Sidewalk lane.
# waypoint01 = map.get_waypoint(vehicle.get_location(), project_to_road=True,
#                               lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
# # Nearest waypoint but specifying OpenDRIVE parameters.
# waypoint02 = map.get_waypoint_xodr(road_id, lane_id, s)
