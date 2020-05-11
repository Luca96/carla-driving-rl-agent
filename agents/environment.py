import os
import math
import enum
import random
import carla
import pygame
import numpy as np

from typing import Optional, TypedDict, Callable, List, Dict, Union
from datetime import datetime

from tensorforce import Environment, Agent

from agents import env_utils, agents
from agents.specifications import Specifications as Specs
from agents.env_features import TemporalFeature
from agents.sensors import Sensor, SensorSpecs

from navigation import Route, RoutePlanner

from tools import misc, utils
from tools.utils import WAYPOINT_DICT, profile
from tools.synchronous_mode import CARLASyncContext


# TODO: add more events
class CARLAEvent(enum.Enum):
    """Available events (callbacks) related to CARLAEnvironment"""
    RESET = 0


# TODO: change name 'train' to 'learn'. Then, one should specify the entire training (episodes, timesteps,
#  saving freq. and so on) just one time, training data should be written in a checkpoint file so that training can
#  be later resumed. -> add the same feature to curriculum-learning training procedure.
class SynchronousCARLAEnvironment(Environment):
    """A TensorForce Environment for the [CARLA driving simulator](https://github.com/carla-simulator/carla).
        - This environment is "synchronized" with the server, meaning that the server waits for a client tick. For a
          detailed explanation of this refer to https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/.
        - Subclass to customize the behaviour of states, actions, sensors, reward function, agent, training loop, etc.

       Requires, you to:
        - Install `pygame`, `opencv`
        - Install the CARLA simulator (version 0.9.8): https://carla.readthedocs.io/en/latest/start_quickstart
        - Install CARLA's Python bindings:
        --> `cd your-path-to/CARLA_0.9.8/PythonAPI/carla/dist/`
        --> Extract `carla-0.9.8-py3.5-YOUR_OS-x86_64.egg` where `YOUR_OS` depends on your OS, i.e. `linux` or `windows`
        --> Create a `setup.py` file within the extracted folder and write the following:
          ```python
          from distutils.core import setup

          setup(name='carla',
                version='0.9.8',
                py_modules=['carla'])
          ```
        --> Install via pip: `pip install -e ~/CARLA_0.9.8/PythonAPI/carla/dist/carla-0.9.8-py3.5-YOUR_OS-x86_64`
        - Run the CARLA simulator from command line: `your-path-to/CARLA_0.9.8/./CarlaUE4.sh` or (CarlaUE4.exe)
        --> To use less resources add these flags: `-windowed -ResX=8 -ResY=8 --quality-level=Low`

        Example usage:
            # Instantiate the environment (run the CARLA simulator before doing this!)
            env = SynchronousCARLAEnvironment(debug=True)

            # Create your own agent
            agent = Agent.create(agent='...',
                                 environment=env,
                                 ...)

            # Training loop (you couldn't use a Runner instead)
            env.train(agent=agent, num_episodes=5, max_episode_timesteps=256, weights_dir=None, record_dir=None)

        Known Issues:
        - TensorForce's Runner is currently not compatible with this environment!
    """
    # States and actions specifications:
    # Actions: throttle or brake, steer, reverse (bool), hand_brake (bool)
    ACTIONS_SPEC = dict(type='float', shape=(4,), min_value=-1.0, max_value=1.0)
    DEFAULT_ACTIONS = np.array([0.0, 0.0, 0.0, 0.0])

    # Vehicle: speed, gear, accelerometer (x, y, z), gyroscope (x, y, z), position (x, y), compass
    VEHICLE_FEATURES_SPEC = dict(type='float', shape=(11,))

    # Road: intersection (bool), junction (bool), speed_limit, traffic_light (presence + state), lane_width,
    #       lane_change, left_lane, right_lane
    ROAD_FEATURES_SPEC = dict(type='float', shape=(10,))

    # TODO: add a loading map functionality (specified or at random) - load_map
    def __init__(self, address='localhost', port=2000, timeout=2.0, image_shape=(150, 200, 3), window_size=(800, 600),
                 vehicle_filter='vehicle.*', sensors: Optional[dict] = None, route_resolution=2.0, fps=30.0,
                 render=True, debug=False):
        """
        :param address: CARLA simulator's id address. Required only if the simulator runs on a different machine.
        :param port: CARLA simulator's port.
        :param timeout: connection timeout.
        :param image_shape: shape of the images observations.
        :param window_size: pygame's window size. Meaningful only if `visualize=True`.
        :param vehicle_filter: use to spawn a particular vehicle (e.g. 'vehicle.tesla.model3') or class of vehicles
            (e.g. 'vehicle.audi.*')
        :param sensors: specifies which sensors should be equipped to the vehicle, better specified by subclassing
            `default_sensors()`.
        :param route_resolution: route planner resolution grain.
        :param fps: maximum framerate, it depends on your computing power.
        :param render: if True a pygame window is shown.
        :param debug: enable to display some useful information about the vehicle. Meaningful only if `render=True`.
        """
        super().__init__()
        env_utils.init_pygame()

        self.timeout = timeout
        self.client = env_utils.get_client(address, port, self.timeout)
        self.world: carla.World = self.client.get_world()

        # TODO: to support 'fixed', 'random', 'list random' map loading its necessary to update the world reference!
        # if isinstance(load_map, str):
        #     self.world: carla.World = self.client.get_world()
        # else:
        #     self.world: carla.World = self.client.load_world(load_map)

        self.map: carla.Map = self.world.get_map()
        self.synchronous_context = None

        # set fix fps:
        self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=False,
            fixed_delta_seconds=1.0 / fps))

        # vehicle
        self.vehicle_filter = vehicle_filter
        self.vehicle: carla.Vehicle = None

        # actions
        self.control: carla.VehicleControl = None
        self.prev_actions = None

        # high-level path planning
        self.route = Route(planner=RoutePlanner(map=self.map, sampling_resolution=route_resolution))
        self.spawn_point = None
        self.destination = None

        # weather
        # TODO: add weather support

        # visualization and debugging stuff
        self.image_shape = image_shape
        self.image_size = (image_shape[1], image_shape[0])
        self.DEFAULT_IMAGE = np.zeros(shape=self.image_shape, dtype=np.float32)
        self.fps = fps
        self.tick_time = 1.0 / self.fps
        self.should_render = render
        self.should_debug = debug
        self.clock = pygame.time.Clock()

        if self.should_render:
            self.window_size = window_size
            self.font = env_utils.get_font(size=13)
            self.display = env_utils.get_display(window_size)

        # variables for reward computation
        self.last_location = None
        self.travelled_distance = 0.0
        self.should_terminate = False
        self.collision_penalty = 0.0
        self.similarity = 0.0
        self.forward_vector = None

        # vehicle sensors suite
        self.sensors_spec = sensors if isinstance(sensors, dict) else self.default_sensors()
        self.sensors = dict()

        # events and callbacks
        self.events: TypedDict[CARLAEvent, Callable] = dict()

    def states(self):
        # TODO: when stacking feature vectors, reshape them into a 2D matrix so that convolutions can be applied!!
        # TODO: consider to include past (more than one) skills, but one-hot encoded!
        return dict(image=dict(shape=self.image_shape),
                    vehicle_features=self.VEHICLE_FEATURES_SPEC,
                    road_features=self.ROAD_FEATURES_SPEC,
                    previous_actions=self.ACTIONS_SPEC)

    def actions(self):
        return self.ACTIONS_SPEC

    def policy_network(self, **kwargs) -> List[dict]:
        """Defines the agent's policy network architecture"""
        raise NotImplementedError

    def reset(self, soft=False) -> dict:
        print('env.reset(soft=True)') if soft else print('env.reset')
        self._reset_world(soft=soft)
        self.trigger_event(event=CARLAEvent.RESET)

        # reset actions
        self.control = carla.VehicleControl()
        self.prev_actions = self.DEFAULT_ACTIONS
        self.should_terminate = False

        observation = env_utils.replace_nans(self._get_observation(sensors_data={}))
        return observation

    def reward(self, actions, time_cost=-1.0, b=2.0, c=2.0, d=2.0):
        """Agent's reward function"""
        # TODO: include a penalty for law compliance: red traffic light...

        # Direction term: alignment of the vehicle's forward vector with the waypoint's forward vector
        speed = utils.speed(self.vehicle)
        similarity = self.similarity

        if 0.8 <= similarity <= 1.0:
            direction_penalty = (speed + 1) * similarity  # speed + 1, to avoid 0 speed
        else:
            direction_penalty = (speed + 1) * abs(similarity) * -d

        # TODO: check
        # also measures the distance from the centre of the road
        if self.control.reverse:
            waypoint_term = self.route.distance_to_next_waypoint() * -b
        else:
            waypoint_term = self.route.distance_to_next_waypoint() * -1

        # Speed-limit compliance:
        speed_limit = self.vehicle.get_speed_limit()

        if speed <= speed_limit:
            speed_penalty = 0.0 if speed > 10.0 else -1.0
        else:
            speed_penalty = c * (speed_limit - speed)

        return time_cost - self.collision_penalty + waypoint_term + direction_penalty + speed_penalty

    def execute(self, actions, record_path: str = None):
        self.prev_actions = actions

        # https://stackoverflow.com/questions/20165492/pygame-window-not-responding-after-a-few-seconds
        pygame.event.get()
        # pygame.event.pump()
        self.clock.tick()

        sensors_data = self.world_step(actions, record_path=record_path)

        reward = self.reward(actions)
        terminal = self.terminal_condition()
        next_state = env_utils.replace_nans(self._get_observation(sensors_data))

        # TODO: penalize remaining distance to destination when terminal=True?

        # Reset collision count
        self.collision_penalty = 0.0

        return next_state, terminal, reward

    def terminal_condition(self, distance_threshold=10.0):
        """Tells whether the episode is terminated or not."""
        return self.should_terminate or \
              (self.route.distance_to_destination(self.vehicle.get_location()) < distance_threshold)

    def close(self):
        print('env.close')
        super().close()

        if self.vehicle:
            self.vehicle.destroy()

        for sensor in self.sensors.values():
            sensor.destroy()

    @profile
    def get_actions(self, agent, states):
        return agent.act(states)

    def train(self, agent: Optional[Agent], num_episodes: int, max_episode_timesteps: int, weights_dir='weights/agents',
              agent_name='carla-agent', record_dir='data/recordings', skip_frames=25):
        record_path = None
        should_record = isinstance(record_dir, str)
        should_save = isinstance(weights_dir, str)

        if agent is None:
            print(f'Using default agent...')
            agent = self.default_agent(max_episode_timesteps=max_episode_timesteps)

        # TODO: introduce callbacks: 'on_episode_start', 'on_episode_end', 'on_update', 'on_record', ..,
        for episode in range(num_episodes):
            states = self.reset()
            total_reward = 0.0

            if should_record:
                record_path = env_utils.get_record_path(base_dir=record_dir)
                print(f'Recording in {record_path}.')

            with self.synchronous_context:
                self.skip(num_frames=skip_frames)
                t0 = datetime.now()

                for i in range(max_episode_timesteps):
                    actions = agent.act(states)
                    # actions = self.get_actions(agent, states)
                    states, terminal, reward = self.execute(actions, record_path=record_path)

                    total_reward += reward
                    terminal = terminal or (i == max_episode_timesteps - 1)

                    if agent.observe(reward, terminal):
                        print(f'{i + 1}/{max_episode_timesteps} -> update performed.')

                    if terminal:
                        elapsed = str(datetime.now() - t0).split('.')[0]
                        print(f'Episode-{episode} completed in {elapsed}, total_reward: {round(total_reward, 2)}\n')
                        break

            if should_save:
                env_utils.save_agent(agent, agent_name, directory=weights_dir)
                print('Agent saved.')

    def default_sensors(self) -> dict:
        """Returns a predefined dict of sensors specifications"""
        return dict(imu=SensorSpecs.imu(),
                    collision=SensorSpecs.collision_detector(callback=self.on_collision),
                    camera=SensorSpecs.rgb_camera(position='top',
                                                  image_size_x=self.image_size[0], image_size_y=self.image_size[1],
                                                  sensor_tick=self.tick_time))

    def default_agent(self, **kwargs) -> Agent:
        """Returns a predefined agent for this environment"""
        raise NotImplementedError('Implement this to define your own default agent!')

    def on_collision(self, event: carla.CollisionEvent, penalty=1000.0, max_impulse=400.0):
        impulse = math.sqrt(utils.vector_norm(event.normal_impulse))
        impulse = min(impulse, max_impulse)
        actor_type = event.other_actor.type_id
        print(f'Collision(impulse={round(impulse, 2)}, actor={actor_type})')

        if 'pedestrian' in actor_type:
            self.collision_penalty += max(penalty * impulse, penalty)
            self.should_terminate = True

        elif 'vehicle' in actor_type:
            self.collision_penalty += max(penalty / 2 * impulse, penalty)
            self.should_terminate = True
        else:
            self.collision_penalty += penalty * impulse
            self.should_terminate = False

    def register_event(self, event: CARLAEvent, callback):
        """Registers a given [callback] to a specific [event]"""
        assert isinstance(event, CARLAEvent)
        assert callable(callback)

        callbacks = self.events.get(event, [])
        callbacks.append(callback)
        self.events[event] = callbacks

    def trigger_event(self, event: CARLAEvent, **kwargs):
        """Cause the call of every callback registered for event [event]"""
        print(f'Event {str(event)} triggered.')
        for callback in self.events.get(event, []):
            callback(**kwargs)

    def render(self, sensors_data: dict):
        """Renders sensors' output"""
        image = sensors_data['camera']
        env_utils.display_image(self.display, image, window_size=self.window_size)

    def debug(self, actions):
        env_utils.display_text(self.display, self.font, text=self.debug_text(actions), origin=(16, 12),
                               offset=(0, 16))

    def debug_text(self, actions):
        return ['%d FPS' % self.clock.get_fps(),
                '',
                'Throttle: %.2f' % self.control.throttle,
                'Steer: %.2f' % self.control.steer,
                'Brake: %.2f' % self.control.brake,
                'Reverse: %s' % ('T' if self.control.reverse else 'F'),
                'Hand brake: %s' % ('T' if self.control.hand_brake else 'F'),
                'Gear: %s' % {-1: 'R', 0: 'N'}.get(self.control.gear),
                '',
                'Speed %.1f km/h' % utils.speed(self.vehicle),
                'Similarity %.2f' % self.similarity,
                '',
                'Reward: %.2f' % self.reward(actions),
                'Collision penalty: %.2f' % self.collision_penalty]

    def skip(self, num_frames=10):
        """Skips the given amount of frames"""
        for _ in range(num_frames):
            self.synchronous_context.tick(timeout=self.timeout)

        if num_frames > 0:
            print(f'Skipped {num_frames} frames.')

    def control_to_actions(self, control: carla.VehicleControl):
        raise NotImplementedError("Implement only if needed for pretraining.")

    def before_world_step(self):
        """Callback: called before world.tick()"""
        pass

    def after_world_step(self, sensors_data: dict):
        """Callback: called after world.tick()."""
        self._update_env_state()

    def on_sensors_data(self, data: dict) -> dict:
        """Callback. Triggers when a world's 'tick' occurs, meaning that data from sensors are been collected because a
        simulation step of the CARLA's world has been completed.
            - Use this method to preprocess sensors' output data for: rendering, observation, ...
        """
        data['camera'] = self.sensors['camera'].convert_image(data['camera'])
        return data

    # @profile
    def world_step(self, actions, record_path: str = None):
        """Applies the actions to the vehicle, and updates the CARLA's world"""
        # [pre-tick updates] Apply control to update the vehicle
        self.actions_to_control(actions)
        self.vehicle.apply_control(self.control)

        self.before_world_step()

        # Advance the simulation and wait for sensors' data.
        data = self.synchronous_context.tick(timeout=self.timeout)
        data = self.on_sensors_data(data)

        # [post-tick updates] Update world-related stuff
        self.after_world_step(data)

        # Draw and debug:
        if self.should_render:
            self.render(sensors_data=data)

            if self.should_debug:
                self.debug(actions)

            pygame.display.flip()

            if isinstance(record_path, str):
                env_utils.pygame_save(self.display, record_path)

        return data

    def _reset_world(self, soft=False):
        # init actor
        if not soft:
            spawn_point = env_utils.random_spawn_point(self.map)
        else:
            spawn_point = self.spawn_point

        if self.vehicle is None:
            blueprint = env_utils.random_blueprint(self.world, actor_filter=self.vehicle_filter)
            self.vehicle: carla.Vehicle = env_utils.spawn_actor(self.world, blueprint, spawn_point)

            self._create_sensors()
            self.synchronous_context = CARLASyncContext(self.world, self.sensors, fps=self.fps)
        else:
            # TODO: in order to respawn the vehicle its needed to: respawn sensors because they are attached to it,
            #  update every component (e.g. planner) that references the vehicle, and so on...
            self.vehicle.apply_control(carla.VehicleControl())
            self.vehicle.set_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))
            self.vehicle.set_transform(spawn_point)

        self.spawn_point = spawn_point
        self.last_location: carla.Location = spawn_point.location
        self.destination: carla.Location = env_utils.random_spawn_point(self.map,
                                                                        different_from=spawn_point.location).location
        # plan path
        self.route.plan(origin=self.spawn_point.location, destination=self.destination)

        # reset reward variables
        self.travelled_distance = 0.0
        self.collision_penalty = 0.0
        self.should_terminate = False
        self._update_waypoint_similarity()

    def _update_env_state(self):
        self._update_target_waypoint()
        self._update_waypoint_similarity()
        self._update_travelled_distance()

    def _update_target_waypoint(self):
        self.route.update_next_waypoint(location=self.vehicle.get_location())

    def _update_waypoint_similarity(self):
        self.forward_vector = self.vehicle.get_transform().get_forward_vector()
        self.similarity = utils.cosine_similarity(self.forward_vector,
                                                  self.route.next.waypoint.transform.get_forward_vector())

    def _update_travelled_distance(self):
        location1 = self.last_location
        location2 = self.vehicle.get_location()

        self.travelled_distance += misc.compute_distance(location1, location2)
        self.last_location = location2

    def actions_to_control(self, actions):
        """Specifies the mapping between an actions vector and the vehicle's control."""
        # throttle and brake are mutual exclusive:
        self.control.throttle = float(actions[0]) if actions[0] > 0 else 0.0
        self.control.brake = float(-actions[0]) if actions[0] < 0 else 0.0

        self.control.steer = float(actions[1])
        self.control.reverse = bool(actions[2] > 0)

        # hand-brake active only if throttle > 0 and reverse is False
        if self.control.throttle > 0 and self.control.reverse:
            self.control.hand_brake = bool(actions[3] > 0)

    def _get_observation(self, sensors_data: dict) -> dict:
        image = sensors_data.get('camera', self.DEFAULT_IMAGE)

        if image.shape != self.image_shape:
            image = env_utils.resize(image, size=self.image_size)

        # Normalize image's pixels value to -1, +1
        observation = dict(image=(2 * image - 255.0) / 255.0,
                           vehicle_features=self._get_vehicle_features(),
                           road_features=self._get_road_features(),
                           previous_actions=self.prev_actions)
        return observation

    def _get_vehicle_features(self):
        t = self.vehicle.get_transform()

        imu_sensor = self.sensors['imu']
        gyroscope = imu_sensor.gyroscope
        accelerometer = imu_sensor.accelerometer

        return [
            utils.speed(self.vehicle),
            self.vehicle.get_control().gear,
            # Accelerometer:
            accelerometer[0],
            accelerometer[1],
            accelerometer[2],
            # Gyroscope:
            gyroscope[0],
            gyroscope[1],
            gyroscope[2],
            # Location
            t.location.x,
            t.location.y,
            # Compass:
            math.radians(imu_sensor.compass)]

    def _get_road_features(self):
        waypoint = self.map.get_waypoint(self.vehicle.get_location())
        speed_limit = self.vehicle.get_speed_limit()
        is_at_traffic_light = self.vehicle.is_at_traffic_light()

        if is_at_traffic_light:
            traffic_light_state = self.vehicle.get_traffic_light_state()
        else:
            traffic_light_state = carla.TrafficLightState.Unknown

        return [float(waypoint.is_intersection),
                float(waypoint.is_junction),
                waypoint.lane_width,
                math.log2(speed_limit),
                # Traffic light:
                float(is_at_traffic_light),
                WAYPOINT_DICT['traffic_light'][traffic_light_state],
                # Lane:
                WAYPOINT_DICT['lane_type'][waypoint.lane_type],
                WAYPOINT_DICT['lane_change'][waypoint.lane_change],
                WAYPOINT_DICT['lane_marking_type'][waypoint.left_lane_marking.type],
                WAYPOINT_DICT['lane_marking_type'][waypoint.right_lane_marking.type]]

    def _create_sensors(self):
        for name, args in self.sensors_spec.items():
            kwargs = args.copy()
            sensor = Sensor.create(sensor_type=kwargs.pop('type'), parent_actor=self.vehicle, **kwargs)

            if name == 'world':
                raise ValueError(f'Cannot name a sensor `world` because is reserved.')

            self.sensors[name] = sensor


# -------------------------------------------------------------------------------------------------
# -- Base Environment
# -------------------------------------------------------------------------------------------------

class CARLABaseEnvironment(Environment):

    def __init__(self, max_timesteps: int, address='localhost', port=2000, timeout=2.0, image_shape=(150, 200, 3),
                 window_size=(800, 600), vehicle_filter='vehicle.*', sensors_spec: Optional[dict] = None, fps=30.0,
                 render=True, debug=True, spawning: dict = None, destination: carla.Location = None, use_planner=True):
        assert isinstance(max_timesteps, int)
        assert isinstance(spawning, dict)

        super().__init__()
        env_utils.init_pygame()

        self.max_timesteps = max_timesteps
        self.timeout = timeout
        self.client = env_utils.get_client(address, port, self.timeout)
        self.world: carla.World = self.client.get_world()
        self.synchronous_context = None

        # Map
        self.map: carla.Map = self.world.get_map()

        # set fix fps:
        self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=False,
            fixed_delta_seconds=1.0 / fps))

        # Vehicle
        self.vehicle_filter = vehicle_filter
        self.vehicle: carla.Vehicle = None
        self.control = carla.VehicleControl()

        # Weather

        # Spawning and destination
        if 'points' in spawning:
            self.spawn_points = spawning.get('points')
            self.spawn_index = -1
            self.spawn_point = None
            self.spawn_type = spawning.get('kind', 'sequential')

            if self.spawn_type == 'random':
                self.spawn_type = 'random_list'
            else:
                assert self.spawn_type == 'sequential'

            assert len(self.spawn_points) > 0
            assert env_utils.all_instances_of(self.spawn_points, kind=carla.Transform)

        elif 'point' in spawning:
            self.spawn_point = spawning.get('point')
            self.spawn_type = spawning.get('kind', 'fixed')

            assert isinstance(self.spawn_point, carla.Transform)
            assert self.spawn_type in ['random', 'fixed']
        else:
            raise ValueError('Dict [spawning] must have either "points" or "point" attribute.')

        self.destination = destination
        self.random_destination = True if destination is None else False

        # Path-planning:
        self.use_planner = use_planner
        if self.use_planner:
            self.route = Route(planner=RoutePlanner(map=self.map, sampling_resolution=2.0))
        else:
            self.route = None

        # Visualization and Debugging
        self.image_shape = image_shape
        self.image_size = (image_shape[1], image_shape[0])
        self.fps = fps
        self.tick_time = 1.0 / self.fps
        self.should_render = render
        self.should_debug = debug
        self.clock = pygame.time.Clock()

        if self.should_render:
            self.window_size = window_size
            self.font = env_utils.get_font(size=13)
            self.display = env_utils.get_display(window_size)

        # vehicle sensors suite
        self.sensors_spec = sensors_spec if isinstance(sensors_spec, dict) else self.default_sensors()
        self.sensors = dict()

        # events and callbacks
        self.events: Dict[CARLAEvent, Callable] = dict()

    def states(self):
        raise NotImplementedError

    def actions(self):
        raise NotImplementedError

    def max_episode_timesteps(self):
        return self.max_timesteps

    def policy_network(self, **kwargs) -> List[dict]:
        """Defines the agent's policy network architecture"""
        raise NotImplementedError

    def reset(self) -> dict:
        print('env.reset')
        self.reset_world()
        self.trigger_event(event=CARLAEvent.RESET)

        self.control = carla.VehicleControl()

        observation = env_utils.replace_nans(self.get_observation(sensors_data={}))
        return observation

    def reward(self, actions, **kwargs):
        """Agent's reward function"""
        raise NotImplementedError

    def execute(self, actions, record_path: str = None):
        pygame.event.get()
        self.clock.tick()

        sensors_data = self.world_step(actions, record_path=record_path)

        reward = self.reward(actions)
        terminal = self.terminal_condition()
        next_state = env_utils.replace_nans(self.get_observation(sensors_data))

        return next_state, terminal, reward

    def terminal_condition(self, **kwargs) -> Union[bool, int]:
        """Tells whether the episode is terminated or not."""
        raise NotImplementedError

    def close(self):
        print('env.close')
        super().close()

        if self.vehicle:
            self.vehicle.destroy()

        for sensor in self.sensors.values():
            sensor.destroy()

    def learn2(self, agent: Agent, num_episodes: int, skip_frames=30):
        """Learning"""
        for episode in range(num_episodes):
            states = self.reset()
            total_reward = 0.0

            with self.synchronous_context:
                self.skip(num_frames=skip_frames)
                t0 = datetime.now()

                for i in range(self.max_episode_timesteps()):
                    actions = agent.act(states)
                    states, terminal, reward = self.execute(actions)

                    total_reward += reward
                    terminal = terminal or (i >= self.max_episode_timesteps() - 1)

                    if agent.observe(reward, terminal):
                        print(f'{i + 1}/{self.max_episode_timesteps()} -> update performed.')

                    if terminal:
                        elapsed = str(datetime.now() - t0).split('.')[0]
                        print(f'Episode-{episode} completed in {elapsed}, total_reward: {round(total_reward, 2)}\n')
                        break

    def learn3(self, agent: Agent, num_episodes: int, save: dict = None, skip_frames=30):
        """Learning"""
        should_save = isinstance(save, dict)

        for episode in range(num_episodes):
            states = self.reset()
            total_reward = 0.0

            with self.synchronous_context:
                self.skip(num_frames=skip_frames)
                t0 = datetime.now()

                for i in range(self.max_episode_timesteps()):
                    actions = agent.act(states)
                    states, terminal, reward = self.execute(actions)

                    total_reward += reward
                    terminal = terminal or (i >= self.max_episode_timesteps() - 1)

                    if agent.observe(reward, terminal):
                        print(f'{i + 1}/{self.max_episode_timesteps()} -> update performed.')

                    if terminal:
                        elapsed = str(datetime.now() - t0).split('.')[0]
                        print(f'Episode-{episode} completed in {elapsed}, total_reward: {round(total_reward, 2)}\n')
                        break

                if should_save and (episode % save['frequency'] == 0):
                    agent.save(directory=save['directory'], filename=save['filename'], format='tensorflow',
                               append=save.get('append', None))

    def evaluate(self, agent: Agent, num_episodes: int, deterministic=True, skip_frames=30) -> List[float]:
        """Evaluation"""
        episodic_rewards = []

        for episode in range(num_episodes):
            states = self.reset()
            episode_reward = 0.0

            with self.synchronous_context:
                self.skip(num_frames=skip_frames)
                t0 = datetime.now()

                for i in range(self.max_episode_timesteps()):
                    actions = agent.act(states, indipendent=True, deterministic=deterministic)
                    states, terminal, reward = self.execute(actions)

                    episode_reward += reward
                    terminal = terminal or (i >= self.max_episode_timesteps() - 1)

                    if terminal:
                        elapsed = str(datetime.now() - t0).split('.')[0]
                        episodic_rewards.append(episode_reward)
                        print(f'Episode-{episode} completed in {elapsed}, total_reward: {round(episode_reward, 2)}\n')
                        break

        return episodic_rewards

    # TODO: provide training statistics, or just let tensorboard to handle them?
    def learn(self, agent: Optional[Agent], num_episodes: int,  weights_dir=None, agent_name='carla-agent',
              record_dir=None, skip_frames=25):
        record_path = None
        should_record = isinstance(record_dir, str)
        should_save = isinstance(weights_dir, str)
        max_episode_timesteps = self.max_episode_timesteps()

        if agent is None:
            print(f'Using default agent...')
            agent = self.default_agent(max_episode_timesteps=max_episode_timesteps)

        for episode in range(num_episodes):
            states = self.reset()
            total_reward = 0.0

            if should_record:
                record_path = env_utils.get_record_path(base_dir=record_dir)
                print(f'Recording in {record_path}.')

            with self.synchronous_context:
                self.skip(num_frames=skip_frames)
                t0 = datetime.now()

                for i in range(max_episode_timesteps):
                    actions = agent.act(states)
                    states, terminal, reward = self.execute(actions, record_path=record_path)

                    total_reward += reward
                    terminal = terminal or (i == max_episode_timesteps - 1)

                    if agent.observe(reward, terminal):
                        print(f'{i + 1}/{max_episode_timesteps} -> update performed.')

                    if terminal:
                        elapsed = str(datetime.now() - t0).split('.')[0]
                        print(f'Episode-{episode} completed in {elapsed}, total_reward: {round(total_reward, 2)}\n')
                        break

            if should_save:
                env_utils.save_agent(agent, agent_name, directory=weights_dir)
                print('Agent saved.')

    def default_sensors(self) -> dict:
        """Returns a predefined dict of sensors specifications"""
        raise NotImplementedError

    def default_agent(self, **kwargs) -> Agent:
        """Returns a predefined agent for this environment"""
        raise NotImplementedError

    def on_collision(self, event: carla.CollisionEvent, **kwargs):
        raise NotImplementedError

    def register_event(self, event: CARLAEvent, callback):
        """Registers a given [callback] to a specific [event]"""
        assert isinstance(event, CARLAEvent)
        assert callable(callback)

        callbacks = self.events.get(event, [])
        callbacks.append(callback)
        self.events[event] = callbacks

    def trigger_event(self, event: CARLAEvent, **kwargs):
        """Cause the call of every callback registered for event [event]"""
        print(f'Event {str(event)} triggered.')
        for callback in self.events.get(event, []):
            callback(**kwargs)

    def render(self, sensors_data: dict):
        """Renders sensors' output"""
        raise NotImplementedError

    def debug(self, actions):
        env_utils.display_text(self.display, self.font, text=self.debug_text(actions), origin=(16, 12),
                               offset=(0, 16))

    def debug_text(self, actions):
        raise NotImplementedError

    def skip(self, num_frames=10):
        """Skips the given amount of frames"""
        for _ in range(num_frames):
            self.synchronous_context.tick(timeout=self.timeout)

        if num_frames > 0:
            print(f'Skipped {num_frames} frames.')

    def control_to_actions(self, control: carla.VehicleControl):
        raise NotImplementedError("Implement only if needed for pretraining.")

    def before_world_step(self):
        """Callback: called before world.tick()"""
        pass

    def after_world_step(self, sensors_data: dict):
        """Callback: called after world.tick()."""
        pass

    def on_sensors_data(self, data: dict) -> dict:
        """Callback. Triggers when a world's 'tick' occurs, meaning that data from sensors are been collected because a
        simulation step of the CARLA's world has been completed.
            - Use this method to preprocess sensors' output data for: rendering, observation, ...
        """
        return data

    def world_step(self, actions, record_path: str = None):
        """Applies the actions to the vehicle, and updates the CARLA's world"""
        # [pre-tick updates] Apply control to update the vehicle
        self.actions_to_control(actions)
        self.vehicle.apply_control(self.control)

        self.before_world_step()

        # Advance the simulation and wait for sensors' data.
        data = self.synchronous_context.tick(timeout=self.timeout)
        data = self.on_sensors_data(data)

        # [post-tick updates] Update world-related stuff
        self.after_world_step(data)

        # Draw and debug:
        if self.should_render:
            self.render(sensors_data=data)

            if self.should_debug:
                self.debug(actions)

            pygame.display.flip()

            if isinstance(record_path, str):
                env_utils.pygame_save(self.display, record_path)

        return data

    def reset_world(self):
        # init actor
        if self.spawn_type == 'random':
            self.spawn_point = env_utils.random_spawn_point(self.map)
        elif self.spawn_type == 'random_list':
            self.spawn_point = random.choice(self.spawn_points)
        elif self.spawn_type == 'sequential':
            self.spawn_index = (self.spawn_index + 1) % len(self.spawn_points)
            self.spawn_point = self.spawn_points[self.spawn_index]

        if self.vehicle is None:
            blueprint = env_utils.random_blueprint(self.world, actor_filter=self.vehicle_filter)
            self.vehicle: carla.Vehicle = env_utils.spawn_actor(self.world, blueprint, self.spawn_point)

            self._create_sensors()
            self.synchronous_context = CARLASyncContext(self.world, self.sensors, fps=self.fps)
        else:
            self.vehicle.apply_control(carla.VehicleControl())
            self.vehicle.set_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))
            self.vehicle.set_transform(self.spawn_point)

        if self.random_destination:
            self.destination = env_utils.random_spawn_point(self.map, different_from=self.spawn_point.location).location

        # plan path
        if self.use_planner:
            self.route.plan(origin=self.spawn_point.location, destination=self.destination)

    def actions_to_control(self, actions):
        """Specifies the mapping between an actions vector and the vehicle's control."""
        raise NotImplementedError

    def get_observation(self, sensors_data: dict) -> dict:
        raise NotImplementedError

    def _create_sensors(self):
        for name, args in self.sensors_spec.items():
            kwargs = args.copy()
            sensor = Sensor.create(sensor_type=kwargs.pop('type'), parent_actor=self.vehicle, **kwargs)

            if name == 'world':
                raise ValueError(f'Cannot name a sensor `world` because is reserved.')

            self.sensors[name] = sensor


# -------------------------------------------------------------------------------------------------
# -- Sync Environment
# -------------------------------------------------------------------------------------------------

# TODO: implement "frame-skip", i.e. concat a frame every N frames so N-1 are skipped between concatenations...
# TODO: store in a file training related stuff so that training can be resumed...
class MyCARLAEnvironment(CARLABaseEnvironment):
    # Control: throttle or brake, steer, reverse
    CONTROL_SPEC = dict(type='float', shape=(3,), min_value=-1.0, max_value=1.0)
    DEFAULT_CONTROL = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Skills: high-level actions
    SKILLS = {0: 'wait', 1: 'brake',
              2: 'steer right', 3: 'steer left',
              4: 'forward', 5: 'forward left', 6: 'forward right',
              7: 'backward', 8: 'backward left', 9: 'backward right'}
    DEFAULT_SKILL = np.array([0.0], dtype=np.float32)
    SKILL_SPEC = dict(type='float', shape=1, min_value=0.0, max_value=len(SKILLS) - 1.0)

    DEFAULT_ACTIONS = dict(control=DEFAULT_CONTROL, skill=DEFAULT_SKILL)

    # Vehicle: speed, acceleration, angular velocity, similarity, distance to waypoint
    VEHICLE_FEATURES_SPEC = dict(type='float', shape=(5,))

    # Road: intersection (bool), junction (bool), speed_limit, traffic_light (presence + state), lane type and change,
    ROAD_FEATURES_SPEC = dict(type='float', shape=(7,))

    def __init__(self, *args, time_horizon=10, radar_shape=(50, 40, 1), **kwargs):
        assert isinstance(radar_shape, tuple)
        super().__init__(*args, **kwargs)
        self.radar_shape = radar_shape

        self.vehicle_obs = TemporalFeature(time_horizon, shape=self.VEHICLE_FEATURES_SPEC['shape'])
        self.skills_obs = TemporalFeature(time_horizon, shape=self.SKILL_SPEC['shape'])
        self.control_obs = TemporalFeature(time_horizon, shape=(4,))
        self.radar_obs = TemporalFeature(time_horizon * 50, shape=(4,))
        self.image_obs = TemporalFeature(time_horizon, shape=self.image_shape[:2], axis=-1)
        self.road_obs = TemporalFeature(time_horizon, shape=self.ROAD_FEATURES_SPEC['shape'])

        # reward computation
        self.collision_penalty = 0.0
        self.should_terminate = False
        self.similarity = 0.0
        self.forward_vector = None

        self.last_actions = self.DEFAULT_ACTIONS
        self.last_location = None
        self.last_travelled_distance = 0.0
        self.total_travelled_distance = 0.0

    def states(self):
        return dict(image=dict(shape=self.image_obs.shape),
                    radar=dict(type='float', shape=self.radar_obs.shape),
                    road=dict(type='float', shape=self.road_obs.shape),
                    vehicle=dict(type='float', shape=self.vehicle_obs.shape),
                    past_control=dict(type='float', shape=self.control_obs.shape),
                    past_skills=dict(type='float', shape=self.skills_obs.shape, min_value=0.0,
                                     max_value=len(self.SKILLS) - 1.0))

    def actions(self):
        return dict(control=self.CONTROL_SPEC, skill=self.SKILL_SPEC)

    def policy_network(self, **kwargs) -> List[dict]:
        features = dict(road=dict(shape=self.road_obs.shape, filters=6, kernel=3, stride=1, layers=4),
                        vehicle=dict(shape=self.vehicle_obs.shape, filters=6, kernel=(3, 4), layers=4),
                        past_control=dict(shape=self.control_obs.shape, filters=6, kernel=(3, 1), layers=4))

        conv_nets = dict(image=dict(filters=22, layers=(2, 5), middle_noise=True, middle_normalization=True),
                         radar=dict(filters=12, reshape=self.radar_shape, layers=(2, 2), activation1='elu', noise=0.0))

        dense_nets = dict(past_skills=dict(units=[24, 30, 30, 30, 24], activation='swish'))  # 24 -> ~3.6k

        # < 0.02ms (agent.act)
        return Specs.network_v4(convolutional=conv_nets, features=features, dense=dense_nets,
                                final=dict(units=[320, 224, 224, 128], activation='swish'))  # 284 -> ~242k

    def reward(self, actions, time_cost=-1, d=2.0, w=10.0, s=2.0, v_max=150.0, d_max=100.0, **kwargs) -> float:
        # Direction term: alignment of the vehicle's forward vector with the waypoint's forward vector
        speed = min(utils.speed(self.vehicle), v_max)
        vel = max(speed / 10.0, 1.0)

        if 0.8 <= self.similarity <= 1.0:
            direction_penalty = vel * self.similarity * (self.action_penalty(actions) + 1)  # ensure coordination
        else:
            direction_penalty = vel * abs(self.similarity) * -d

        # Distance from waypoint (and also lane center)
        waypoint_term = min(self.route.distance_to_next_waypoint(), d_max) * -w

        # Speed-limit compliance:
        speed_limit = self.vehicle.get_speed_limit()

        if speed <= speed_limit:
            speed_penalty = 0.0 if speed >= 10.0 else speed - 10.0
        else:
            speed_penalty = s * (speed_limit - speed)

        # almost bounded [-2250, +60]
        return time_cost - self.collision_penalty + waypoint_term + direction_penalty + speed_penalty

    def reward2(self, actions, s=2.0, max_speed=150.0, **kwargs) -> float:
        speed = min(utils.speed(self.vehicle), max_speed)
        speed_limit = self.vehicle.get_speed_limit()

        if speed > speed_limit:
            speed_compliance = s * (speed - speed_limit)
        else:
            speed_compliance = 0.0

        return self.action_penalty(actions) * (self.last_travelled_distance * self.similarity) - \
            self.collision_penalty - speed_compliance

    def reset(self) -> dict:
        self.last_actions = self.DEFAULT_ACTIONS
        self.should_terminate = False
        self.total_travelled_distance = 0.0
        self.last_travelled_distance = 0.0

        # reset observations (np.copyto() should reuse memory...)
        self.control_obs.reset()
        self.radar_obs.reset()
        self.image_obs.reset()
        self.road_obs.reset()
        self.skills_obs.reset()
        self.vehicle_obs.reset()

        observation = super().reset()

        self.last_location = self.spawn_point.location
        return observation

    def execute(self, actions, record_path: str = None):
        self.last_actions = actions

        next_state, terminal, reward = super().execute(actions, record_path)

        self.collision_penalty = 0.0
        return next_state, terminal, reward

    def get_skill_name(self):
        """Returns skill's name"""
        index = round(self.last_actions['skill'][0])
        return self.SKILLS[index]

    @staticmethod
    def action_penalty(actions, eps=0.05) -> float:
        """Returns the amount of coordination, defined as the number of actions that agree with the skill"""
        skill = round(actions['skill'][0])
        a0, steer, a2 = actions['control']
        num_actions = len(actions['control'])
        throttle = max(a0, 0.0)
        reverse = a2 > 0
        count = 0

        # wait/noop
        if skill == 0:
            count += 1 if throttle > eps else 0

        # brake
        elif skill == 1:
            count += 1 if throttle > eps else 0

        # steer right/left
        elif skill in [2, 3]:
            count += 1 if -eps <= steer <= eps else 0
            count += 1 if throttle > eps else 0

        # forward right/left
        elif skill in [4, 5, 6]:
            count += 1 if reverse else 0
            count += 1 if throttle < eps else 0

            if skill == 4:
                count += 0 if -eps <= steer <= eps else 1
            elif skill == 5:
                count += 1 if steer > -eps else 0
            else:
                count += 1 if steer < eps else 0

        # backward right/left
        elif skill in [7, 8, 9]:
            count += 1 if not reverse else 0
            count += 1 if throttle < eps else 0

            if skill == 7:
                count += 0 if -eps <= steer <= eps else 1
            elif skill == 8:
                count += 1 if steer > -eps else 0
            else:
                count += 1 if steer < eps else 0

        return num_actions - count

    def terminal_condition(self, **kwargs) -> Union[bool, int]:
        if self.should_terminate:
            return 2

        return self.route.distance_to_destination(self.vehicle.get_location()) < 2.0

    def default_sensors(self) -> dict:
        return dict(collision=SensorSpecs.collision_detector(callback=self.on_collision),
                    imu=SensorSpecs.imu(),
                    camera=SensorSpecs.segmentation_camera(position='on-top2', attachment_type='Rigid',
                                                           image_size_x=self.image_size[0],
                                                           image_size_y=self.image_size[1],
                                                           sensor_tick=self.tick_time),
                    radar=SensorSpecs.radar(position='radar', sensor_tick=self.tick_time))

    def default_agent(self, **kwargs) -> Agent:
        return agents.Agents.ppo6(self, self.max_episode_timesteps(), batch_size=kwargs.get('batch_size', 1))

    def on_collision(self, event: carla.CollisionEvent, penalty=1000.0):
        actor_type = event.other_actor.type_id
        print(f'Collision with actor={actor_type})')

        if 'pedestrian' in actor_type:
            self.collision_penalty += penalty
            self.should_terminate = True

        elif 'vehicle' in actor_type:
            self.collision_penalty += penalty / 2.0
            self.should_terminate = True
        else:
            self.collision_penalty += penalty / 100.0
            self.should_terminate = False

    def render(self, sensors_data: dict):
        image = sensors_data['camera']
        env_utils.display_image(self.display, image, window_size=self.window_size)

    def debug_text(self, actions):
        speed_limit = self.vehicle.get_speed_limit()
        speed = utils.speed(self.vehicle)
        distance = self.total_travelled_distance

        if speed > speed_limit:
            speed_text = dict(text='Speed %.1f km/h' % speed, color=(255, 0, 0))
        else:
            speed_text = 'Speed %.1f km/h' % speed

        return ['%d FPS' % self.clock.get_fps(),
                '',
                'Throttle: %.2f' % self.control.throttle,
                'Steer: %.2f' % self.control.steer,
                'Brake: %.2f' % self.control.brake,
                'Reverse: %s' % ('T' if self.control.reverse else 'F'),
                '',
                speed_text,
                'Speed limit %.1f km/h' % speed_limit,
                'Distance travelled %.2f %s' % ((distance / 1000.0, 'km') if distance > 1000.0 else (distance, 'm')),
                '',
                'Similarity %.2f' % self.similarity,
                'Waypoint\'s Distance %.2f' % self.route.distance_to_next_waypoint(),
                '',
                'Reward: %.2f' % self.reward(actions),
                'Collision penalty: %.2f' % self.collision_penalty,
                'Skill (%d) = %s' % (round(self.last_actions['skill'][0]), self.get_skill_name()),
                'Coordination %d' % self.action_penalty(actions)]

    def control_to_actions(self, control: carla.VehicleControl):
        pass

    def on_sensors_data(self, data: dict) -> dict:
        data['camera'] = self.sensors['camera'].convert_image(data['camera'])
        data['radar'] = self.sensors['radar'].convert(data['radar'])
        return data

    def after_world_step(self, sensors_data: dict):
        self._update_env_state()

    def actions_to_control(self, actions):
        actions = actions['control']
        self.control.throttle = float(actions[0]) if actions[0] > 0 else 0.0
        self.control.brake = float(-actions[0]) if actions[0] < 0 else 0.0
        self.control.steer = float(actions[1])
        self.control.reverse = bool(actions[2] > 0)
        self.control.hand_brake = False

    def get_observation(self, sensors_data: dict) -> dict:
        if len(sensors_data.keys()) == 0:
            # sensor_data is empty so, return a default observation
            return dict(image=self.image_obs.default, radar=self.radar_obs.default, vehicle=self.vehicle_obs.default,
                        road=self.road_obs.default, past_control=self.control_obs.default,
                        past_skills=self.skills_obs.default)

        # resize image if necessary
        image = sensors_data['camera']

        if image.shape != self.image_shape:
            image = env_utils.resize(image, size=self.image_size)

        # grayscale image, plus -1, +1 scaling
        image = (2 * env_utils.cv2_grayscale(image) - 255.0) / 255.0
        radar = sensors_data['radar']

        # concat new observations along the temporal axis:
        self.vehicle_obs.append(value=self._get_vehicle_features())
        self.control_obs.append(value=self._control_as_vector())
        self.skills_obs.append(value=self.last_actions['skill'].copy())
        self.road_obs.append(value=self._get_road_features())
        self.image_obs.append(image, depth=True)

        # copy radar measurements
        for i, detection in enumerate(radar):
            self.radar_obs.append(detection)

        # observation
        return dict(image=self.image_obs.data, radar=self.radar_obs.data, vehicle=self.vehicle_obs.data,
                    road=self.road_obs.data, past_control=self.control_obs.data, past_skills=self.skills_obs.data)

    def _control_as_vector(self) -> list:
        return [self.control.throttle, self.control.brake, self.control.steer, float(self.control.reverse)]

    def _get_road_features(self):
        waypoint = self.map.get_waypoint(self.vehicle.get_location())
        speed_limit = self.vehicle.get_speed_limit()
        is_at_traffic_light = self.vehicle.is_at_traffic_light()

        if is_at_traffic_light:
            traffic_light_state = self.vehicle.get_traffic_light_state()
        else:
            traffic_light_state = carla.TrafficLightState.Unknown

        # get current lane type: consider only road (driving) lanes
        if waypoint.lane_type is carla.LaneType.NONE:
            lane_type = 0
        elif waypoint.lane_type is carla.LaneType.Driving:
            lane_type = 1
        else:
            lane_type = 0

        return [waypoint.is_intersection,
                waypoint.is_junction,
                round(speed_limit / 10.0),
                # Traffic light:
                is_at_traffic_light,
                WAYPOINT_DICT['traffic_light'][traffic_light_state],
                # Lane:
                lane_type,
                WAYPOINT_DICT['lane_change'][waypoint.lane_change]]

    def _get_vehicle_features(self):
        imu_sensor = self.sensors['imu']

        # vehicle's acceleration (also considers direction)
        acceleration = env_utils.magnitude(imu_sensor.accelerometer) * env_utils.sign(self.similarity)

        # vehicle's angular velocity
        angular_velocity = env_utils.magnitude(imu_sensor.gyroscope)

        return [utils.speed(self.vehicle) / 10.0,
                acceleration,
                angular_velocity,
                # Target (next) waypoint's features:
                self.similarity,
                self.route.distance_to_next_waypoint()]

    # TODO: move to base class
    def _update_env_state(self):
        if self.use_planner:
            self._update_target_waypoint()
            self._update_waypoint_similarity()

        self._update_travelled_distance()

    def _update_target_waypoint(self):
        self.route.update_next_waypoint(location=self.vehicle.get_location())

    def _update_waypoint_similarity(self):
        self.forward_vector = self.vehicle.get_transform().get_forward_vector()
        self.similarity = utils.cosine_similarity(self.forward_vector,
                                                  self.route.next.waypoint.transform.get_forward_vector())

    # TODO: move to base class
    def _update_travelled_distance(self):
        location1 = self.last_location
        location2 = self.vehicle.get_location()

        self.last_travelled_distance = misc.compute_distance(location1, location2)
        self.total_travelled_distance += self.last_travelled_distance
        self.last_location = location2
