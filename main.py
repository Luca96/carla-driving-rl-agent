import os
import carla
import pygame
import logging
import numpy as np
import tensorflow as tf
import tensorforce

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

        hud = HUD(width, height)
        world = World(client.get_world(), hud, actor_filter=vehicle)
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
                 image_shape=(200, 150, 3), **kwargs):
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


def test_sync_event(agent_builder, timesteps, episodes, vehicle='vehicle.tesla.model3', width=800, height=600,
                    image_shape=(200, 150, 3), **kwargs):
    pygame.init()
    pygame.font.init()
    sync_env = None
    runner = None

    try:
        sync_env = SynchronousCARLAEnvironment(image_shape=image_shape,
                                               window_size=(width, height),
                                               timeout=5.0,
                                               debug=True,
                                               vehicle_filter=vehicle)
        print('== ENV ==')

        agent = agent_builder(sync_env, max_episode_timesteps=timesteps, **kwargs)
        print('== AGENT ==')

        for episode in range(episodes):
            states = sync_env.reset()
            terminal = False
            total_reward = 0.0

            with sync_env.synchronous_context:
                for i in tqdm(range(timesteps), desc=f'E-{episode} [{total_reward}]'):
                    actions = agent.act(states)
                    states, terminal, reward = sync_env.execute(actions)
                    total_reward += reward

                    # # print('reward:', reward)
                    update_performed = agent.observe(reward, terminal)

                    if update_performed:
                        print(f'{i}/{timesteps} -> update performed.')

                    if terminal:
                        break

            print(f'E-{episode} total_reward: {round(total_reward, 2)}')

            # agent.save(directory='weights/agents', filename='carla-agent')
            # print('agent saved.')

        # runner = Runner(agent, sync_env, max_episode_timesteps=timesteps, num_parallel=None)
        # runner.run(num_episodes=episodes)

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
    # TODO: to make Tensorforce work with tensorflow 2.0.1,
    #  comment two lines in '/usr/lib/python3.8/site-packages/tensorflow_core/lite/experimental/microfrontend/python/ops
    #  /audio_microfrontend_op.py'

    # game_loop()

    network = specs.agent_network()
    light_net = specs.agent_light_network()

    num_episodes = 3
    num_timesteps = 256 * 1
    frequency = 64
    batch_size = 256

    agent_args = dict(policy=dict(type='parametrized_distributions',
                                  network=network,
                                  temperature=0.95),

                      optimizer=dict(type='adam', learning_rate=3e-4),

                      objective=specs.obj.policy_gradient(clipping_value=0.2),

                      # update=specs.update(unit='timesteps', batch_size=0, frequency='never'),
                      update=specs.update(unit='timesteps', batch_size=batch_size, frequency=frequency),

                      # Baseline (critic)
                      baseline_policy=dict(type='parametrized_distributions',
                                           distributions=dict(float='gaussian'),
                                           network=network,
                                           temperature=0.99),
                      baseline_optimizer=dict(type='adam', learning_rate=0.0003),
                      baseline_objective=specs.obj.value(value='state', huber_loss=0.1),

                      reward_estimation=dict(discount=1.0,
                                             horizon=100,
                                             estimate_horizon='early',
                                             # estimate_actions=False,
                                             estimate_advantage=True),

                      # preprocessing=dict(previous_actions=dict(type='deltafier'),
                      #                    reward=dict(type='deltafier')),


                      # preprocessing=dict(image=dict(type='image', height=60, width=60, grayscale=True)),
                      # preprocessing=dict(image=dict(type='image', grayscale=True)),

                      # preprocessing=dict(type='deltafier'),

                      # recorder=dict(directory='data/traces'),

                      exploration=0.05,  # exp_decay(steps=50, rate=0.5),
                      entropy_regularization=0.1)

    # test_new_env(AgentConfigs.tensorforce2, num_timesteps, num_episodes, **agent_args)
    test_sync_event(AgentConfigs.tensorforce2, num_timesteps, num_episodes, **agent_args)

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

# https://carla.readthedocs.io/en/latest/core_world/
# Find and attach a sensor:
# collision_sensor_bp = blueprint_library.find('sensor.other.collision')
# camera_bp = blueprint_library.find('sensor.camera.rgb')
# camera = worlds.spawn_actor(camera_bp, relative_transform, attach_to=my_vehicle)
# camera.listen(lambda image: image.save_to_disk('output/%06d.png' % image.frame))

# Find actors:
# actor_list = worlds.get_actors()
# # Find an actor by id.
# actor = actor_list.find(id)
# # Print the location of all the speed limit signs in the worlds.
# for speed_sign in actor_list.filter('traffic.speed_limit.*'):
#     print(speed_sign.get_location())

# Destroy actor:
# destroyed_successfully = actor.destroy()

# Get the traffic light affecting a vehicle: Traffic lights will only affect a vehicle if the light is red.
# if vehicle_actor.is_at_traffic_light():
#     traffic_light = vehicle_actor.get_traffic_light()

# Change a red traffic light to green
# if traffic_light.get_state() == carla.TrafficLightState.Red:
#     traffic_light.set_state(carla.TrafficLightState.Green)

# vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1.0))
# vehicle.apply_physics_control(
#     carla.VehiclePhysicsControl(max_rpm=5000.0, center_of_mass=carla.Vector3D(0.0, 0.0, 0.0),
#                                 torque_curve=[[0, 400], [5000, 400]]))

# --> vehicle.get_traffic_light()
# --> vehicle.get_speed_limit()

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

# https://carla.readthedocs.io/en/latest/cameras_and_sensors/
# # Find the blueprint of the sensor.
# blueprint = worlds.get_blueprint_library().find('sensor.camera.rgb')
# # Modify the attributes of the blueprint to set image resolution and field of view.
# blueprint.set_attribute('image_size_x', '1920')
# blueprint.set_attribute('image_size_y', '1080')
# blueprint.set_attribute('fov', '110')
# # Set the time in seconds between sensor captures
# blueprint.set_attribute('sensor_tick', '1.0')
# # Provide the position of the sensor relative to the vehicle.
# transform = carla.Transform(carla.Location(x=0.8, z=1.7))
# # Tell the worlds to spawn the sensor, don't forget to attach it to your vehicle actor.
# sensor = worlds.spawn_actor(blueprint, transform, attach_to=my_vehicle)
# # Subscribe to the sensor stream by providing a callback function, this function is
# # called each time a new image is generated by the sensor.
# sensor.listen(lambda data: do_something(data))

# Currently available sensors:
# sensor.camera.rgb  -> produces carla.Image objects (default 800x600, fov=90, sensor_tick=0, shutter_speed=60)

# sensor.camera.depth -> carla.Image (default 800x600, fov=90, sensor_tick=0)
# Decode the distance in meters from a depth image:
# normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
# in_meters = 1000 * normalized

# sensor.camera.semantic_segmentation -> carla.Image (800x600, 90, 0), tag information are in the red channel.
# Tags <value, tag, color>:
# 0  Unlabeled    (0, 0, 0)
# 1  Building     (70, 70, 70)
# 2  Fence        (190, 153, 153)
# 3  Other        (250, 170, 160)
# 4  Pedestrian   (220, 20, 60)
# 5  Pole         (153, 153, 153)
# 6  Road line    (157, 234, 50)
# 7  Road         (128, 64, 128)
# 8  Sidewalk     (244, 35, 232)
# 9  Vegetation   (107, 142, 35)
# 10 Car          (0, 0, 142)
# 11 Wall         (102, 102, 156)
# 12 Traffic sign (220, 220, 0)

# sensor.lidar.ray_cast
# sensor.other.collision -> carla.CollisionEvent(frame, timestamp, transform, actor, other_actor, normal_impulse)
# sensor.other.lane_invasion -> carla.LaneInvasionEvent(frame, timestamp, transform, actor, crossed_lane_markings)
# sensor.other.gnss -> carla.GnssMeasurement(frame, timestamp, transform, latitude, longitude, altitude)

# sensor.other.obstacle -> carla.ObstacleDetectionEvent(//, //, //, actor, other_actor, distance)
# sensor blueprint attributes: distance, hit_radius, only_dynamics, debug_linetrace, sensor_tick

# sensor.other.imu: let the user access it's accelerometer, gyroscope and compass
# -> carla.IMUMeasurement(//, //, //, accelerometer, gyroscope, compass)
