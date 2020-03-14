import os
import carla
import pygame
import logging
import tensorflow as tf

from tensorforce import Runner, Environment, Agent
from tensorforce.agents import PPOAgent, A2CAgent

from agents import AgentConfigs
from agents.learn import CarlaEnvironment, CarlaCompressImageEnv
from agents.network import baseline, print_network
from agents.specifications import Specifications as specs

from worlds import World
from worlds.controllers import KeyboardController
from worlds.debug import HUD


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


class MicroMock(object):
    """Creates generic objects with fields given as named arguments."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def game_loop(vehicle='vehicle.audi.*', width=800, height=600):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = get_client(timeout=2.0)
        display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(width, height)
        world = World(client.get_world(), hud, actor_filter=vehicle)
        controller = KeyboardController(world)

        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(60)

            if controller.parse_events(client, world, clock, training=False):
                return

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


# TODO: remove
def test_learning_env(timesteps: int, episodes: int, agent_type='a2c', vehicle='vehicle.tesla.model3', width=800, height=600):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = get_client()
        display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(width, height)
        world = World(client.get_world(), hud, actor_filter=vehicle)
        controller = KeyboardController(world)

        environment = CarlaEnvironment(world, display, image_shape=(width // 4, height // 4, 3))
        print('== ENV ==')

        # agent = Agent.create(
        #     agent=agent_type, environment=environment, max_episode_timesteps=timesteps,
        #     # Network specification
        #     # network=dict(type='auto', size=64, depth=2, final_size=256, final_depth=2, internal_rnn=10),
        #     network=dict(type='auto', size=64, depth=2, final_size=256, final_depth=2, internal_rnn=False),
        #     # Optimization
        #     batch_size=1, learning_rate=1e-3, subsampling_fraction=0.2,
        #     optimization_steps=5,
        #     # Reward estimation
        #     likelihood_ratio_clipping=0.2, discount=0.99, estimate_terminal=False,
        #     # Critic
        #     critic_network='auto',
        #     # critic_optimizer=dict(optimizer='adam', multi_step=10, learning_rate=1e-3),
        #     critic_optimizer=dict(optimizer='adam', multi_step=1, learning_rate=1e-3),
        #     # Pre-processing
        #     # preprocessing=dict(reward=dict(type='deltafier')),
        #     # preprocessing=dict(image=dict(type='image', grayscale=True),
        #     #                    reward=dict(type='deltafier')),
        #
        #     # preprocessing=dict(state=[dict(type='image', height=100, width=100, grayscale=True)],
        #     #                    reward=dict(type='deltafier')),
        #     # Exploration
        #     exploration=0.0, variable_noise=0.0,  # exp_decay(steps=500, rate=0.5)
        #     # Regularization
        #     l2_regularization=0.0, entropy_regularization=0.0,
        #     # TensorFlow etc
        #     name='agent', device=None, parallel_interactions=1, seed=None, execution=None, saver=None,
        #     # summarizer=tensorboard_summarizer(), recorder=None
        # )
        #
        agent = Agent.create(agent='random',
                             environment=environment,
                             max_episode_timesteps=timesteps)

        print('== AGENT ==')
        # print('\tactions_spec:', agent.actions_spec)
        # print('\tepisodes:', agent.episodes)
        # print('\texperience_size:', agent.experience_size)
        # print('\tsummaries:', agent.get_available_summaries())
        # print('\tmax_episode_timesteps:', agent.max_episode_timesteps)
        # print('\tmodel:', agent.model)
        # print('\tstates_spec:', agent.states_spec)
        # print('\tspec:', agent.spec)
        # print('\ttimesteps:', agent.timesteps)
        # print_object(agents, message='agents')

        runner = Runner(agent, environment, max_episode_timesteps=timesteps, num_parallel=None)
        # runner.run(num_episodes=8, save_best_agent='weights')
        runner.run(num_episodes=episodes)

        # for episode in range(episodes):
        #     print('Epoch ' + str(episode))
        #     states = environment.reset()
        #     terminal = False
        #     total_reward = 0.0
        #
        #     for i in range(episode_timesteps):
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
        runner.close()

    finally:
        if world and world.recording_enabled:
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


def test_new_env(agent_builder, timesteps, episodes, vehicle='vehicle.tesla.model3', width=800, height=600, **kwargs):
    pygame.init()
    pygame.font.init()

    try:
        environment = CarlaEnvironment(client=get_client(),
                                       image_shape=(200, 150, 3),
                                       actor_filter=vehicle,
                                       max_fps=20,
                                       window_size=(800, 600))
        print('== ENV ==')

        # controller = KeyboardController(environment.world)

        agent = agent_builder(environment, max_episode_timesteps=timesteps, **kwargs)
        print('== AGENT ==')

        runner = Runner(agent, environment, max_episode_timesteps=timesteps, num_parallel=None)
        runner.run(num_episodes=episodes, save_best_agent='weights/agents')
        runner.close()

    finally:
        pygame.quit()


def gym_env_test(env='CartPole-v1', episodes=30):
    # Create an OpenAI-Gym environment
    environment = Environment.create(environment='gym', level=env, visualize=True)

    # Create a PPO agent
    # agent = Agent.create(
    #     agent='ppo', environment=environment,
    #     # Automatically configured network
    #     network='auto',
    #     # Optimization
    #     batch_size=10, update_frequency=2, learning_rate=1e-3, subsampling_fraction=0.2,
    #     optimization_steps=5,
    #     # Reward estimation
    #     likelihood_ratio_clipping=0.2, discount=0.99, estimate_terminal=False,
    #     # Critic
    #     critic_network='auto',
    #     critic_optimizer=dict(optimizer='adam', multi_step=10, learning_rate=1e-3),
    #     # Preprocessing
    #     preprocessing=None,
    #     # Exploration
    #     exploration=0.0, variable_noise=0.0,
    #     # Regularization
    #     l2_regularization=0.0, entropy_regularization=0.0,
    #     # TensorFlow etc
    #     name='agent', device=None, parallel_interactions=1, seed=None, execution=None, saver=None,
    #     summarizer=None, recorder=None
    # )

    network_spec = dict(type='auto', depth=2, final_depth=2, internal_rnn=5)

    policy_spec = dict(type='parametrized_distributions',
                       # distributions=dict(type='categorical'),
                       # distributions=dict(type='gaussian'),
                       # temperature=0.1,
                       network=network_spec)

    # optimizer_spec = dict(type='optimizing_step', optimizer=dict(type='natural_gradient', learning_rate=2e-2))
    # optimizer_spec = dict(type='natural_gradient', learning_rate=3e-3)

    # objective_spec = dict(type='det_policy_gradient')
    objective_spec = dict(type='policy_gradient', clipping_value=0.2)
    # objective_spec = dict(type='plus',
    #                       objective1=dict(type='policy_gradient', clipping_value=0.2),
    #                       objective2=dict(type='value', value='action', huber_loss=0.5))

    update_spec = dict(unit='timesteps', batch_size=2)

    # reward_spec = dict(horizon=20, discount=0.99, estimate_actions=True, estimate_advantage=True)
    reward_spec = dict(horizon=20, discount=0.99)
    # reward_spec = dict(horizon=10, discount=0.99)

    agent = Agent.create(agent='tensorforce',
                         environment=environment,
                         policy=policy_spec,
                         objective=objective_spec,
                         optimizer='adadelta',
                         update=update_spec,
                         reward_estimation=reward_spec,
                         exploration=0.0,
                         entropy_regularization=0.1)

    # Initialize the runner
    runner = Runner(agent=agent, environment=environment)

    # Start the runner
    runner.run(num_episodes=episodes)
    runner.close()


if __name__ == '__main__':
    # TODO: to make Tensorforce work with tensorflow 2.0.1,
    #  comment two lines in '/usr/lib/python3.8/site-packages/tensorflow_core/lite/experimental/microfrontend/python/ops
    #  /audio_microfrontend_op.py'

    # agent = Agent.create(...
    # recorder = dict(
    #     directory='data/traces',
    #     frequency=100  # record a traces file every 100 episodes
    # ), ...
    # )
    # ...
    # agent.close()
    #
    # # Pretrain agent on recorded traces
    # agent = Agent.create(...)
    # agent.pretrain(
    #     directory='data/traces',
    #     num_iterations=100  # perform 100 update iterations on traces (more configurations possible)
    # )

    # print_network(baseline)

    # TF_ARGS = dict(batch_size=1, horizon=0)
    PPO_CONF = dict(batch_size=1, optimization_steps=1, discount=0.99,
                    network=dict(type='auto', size=64, depth=3, final_size=256, final_depth=1, internal_rnn=False),
                    critic_network='auto', critic_optimizer=dict(optimizer='adam', multi_step=1, learning_rate=1e-3),
                    # preprocessing=dict(image=dict(type='image', grayscale=True),
                    #                    reward=dict(type='deltafier')),

                    preprocessing=dict(state=[dict(type='image', grayscale=True)],
                                       reward=dict(type='deltafier')),
                    )

    # test_learning_env(agent_type='ppo', timesteps=120, episodes=10)
    # test_new_env(AgentConfigs.tensorforce, timesteps=120, episodes=10, **TF_ARGS)
    # test_new_env(AgentConfigs.ppo, timesteps=100, episodes=10, **PPO_CONF)
    # game_loop()
    # gym_env_test(env='LunarLander-v2', episodes=100)
    # gym_env_test(env='CarRacing-v0', episodes=100)

    # test_new_env(AgentConfigs.tensorforce2,
    #              timesteps=256,
    #              episodes=10,
    #              policy=specs.policy(distributions='gaussian',
    #                                  # network=specs.auto_network(final_depth=2, final_size=256, internal_rnn=False),
    #                                  network=specs.complex_network(networks=[
    #                                      specs.conv_network(inputs='image', layers=5, dilation=2, dropout=0.2,
    #                                                         output='image_out'),
    #                                      specs.dense_network(inputs='vehicle_features', layers=2, units=32, dropout=0.2,
    #                                                          output='vehicle_out'),
    #                                      specs.dense_network(inputs='road_features', layers=2, units=24, dropout=0.2,
    #                                                          output='road_out'),
    #                                      specs.dense_network(inputs='previous_actions', layers=1, units=16, dropout=0.2,
    #                                                          output='actions_out')],
    #                                      layers=2,
    #                                      units=200),
    #                                  temperature=0.1),
    #              optimizer=specs.opt.subsampling_step(optimizer=specs.opt.multi_step(optimizer='adam', num_steps=2),
    #                                                   fraction=0.2),
    #              objective=specs.obj.policy_gradient(clipping_value=0.2),
    #              update=specs.update(unit='timesteps', batch_size=32),
    #              reward_estimation=specs.reward_estimation(horizon=30, discount=0.99, estimate_horizon='late',
    #                                                        estimate_actions=False, estimate_advantage=False),
    #              exploration=0.1,  # exp_decay(steps=50, rate=0.5),
    #              entropy_regularization=0.05)

    network = specs.complex_network(networks=[
                                         specs.conv_network(inputs='image', layers=5, stride=2, pool=None, dropout=0.2,
                                                            output='image_out'),
                                         specs.dense_network(inputs='vehicle_features', layers=2, units=32, dropout=0.2,
                                                             output='vehicle_out'),
                                         specs.dense_network(inputs='road_features', layers=2, units=24, dropout=0.2,
                                                             output='road_out'),
                                         specs.dense_network(inputs='previous_actions', layers=1, units=16, dropout=0.2,
                                                             output='actions_out')],
                                    layers=2,
                                    units=200)

    num_episodes = 10
    num_timesteps = 256 * 2
    frequency = 64
    batch_size = num_timesteps

    test_new_env(AgentConfigs.tensorforce2,
                 timesteps=num_timesteps,
                 episodes=num_episodes,

                 policy=specs.policy(distributions='gaussian',
                                     network=network,
                                     temperature=0.99),

                 optimizer=specs.opt.subsampling_step(optimizer=dict(type='adam', learning_rate=3e-4),
                                                      fraction=0.25),

                 objective=specs.obj.policy_gradient(clipping_value=0.2),

                 update=specs.update(unit='timesteps', batch_size=batch_size, frequency=frequency),

                 # Baseline (critic)
                 # baseline_policy=specs.policy(distributions='gaussian', network=network),
                 # baseline_optimizer=dict(type='adam', learning_rate=0.001),
                 # baseline_objective=specs.obj.value(value='action', huber_loss=0.0),
                 #
                 reward_estimation=dict(horizon=200,
                                        discount=0.997,  # 0.997^200 ~ 0.5
                                        # estimate_horizon='early',
                                        # estimate_actions=True,
                                        estimate_advantage=True
                                        ),

                 exploration=0.1 * 0,  # exp_decay(steps=50, rate=0.5),
                 entropy_regularization=0.05)

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
