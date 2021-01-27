import os
import gym
import time
import enum
import random
import carla
import pygame
import numpy as np

from gym import spaces
from typing import Callable, Dict, Union, List
from pygame.constants import K_q, K_UP, K_w, K_LEFT, K_a, K_RIGHT, K_d, K_DOWN, K_s, K_SPACE, K_ESCAPE, KMOD_CTRL

from rl import utils as rl_utils

from rl.environments.carla import env_utils
from rl.environments.carla.sensors import Sensor, SensorSpecs

from rl.environments.carla.navigation.behavior_agent import BehaviorAgent
from rl.environments.carla.navigation import Route, RoutePlanner, RoadOption

from rl.environments.carla.tools import misc, utils
from rl.environments.carla.tools.utils import WAYPOINT_DICT
from rl.environments.carla.tools.synchronous_mode import CARLASyncContext


class CARLAEvent(enum.Enum):
    """Available events (callbacks) related to CARLAEnvironment"""
    RESET = 0
    ON_COLLISION = 1
    OUT_OF_LANE = 2


# -------------------------------------------------------------------------------------------------
# -- Base Class and Wrappers
# -------------------------------------------------------------------------------------------------

# TODO: use gym register API to make these environments available to gym.make(...)
class CARLABaseEnvironment(gym.Env):
    """Base extendable environment for the CARLA driving simulator"""

    def __init__(self, address='localhost', port=2000, timeout=5.0, image_shape=(150, 200, 3), window_size=(800, 600),
                 vehicle_filter='vehicle.tesla.model3', fps=30.0, render=True, debug=True, spawn: dict = None,
                 ignore_traffic_light=True, path: dict = None, town: str = None,
                 weather=carla.WeatherParameters.ClearNoon, skip_frames=30):
        """Arguments:
            - path: dict =
                - origin: dict(carla.Transform or 'point' or 'points', 'type': [fixed, route, map] or [random,
                               sequential])
                - destination: dict(carla.Transform or 'point' or 'points', 'type': [fixed, map] or [random,
                               sequential]))
                - length: int
                - use_planner: bool
                - sampling_resolution: float

            - spawn: dict(vehicle_filter: str, pedestrian_filter: str, pedestrians: int, vehicles: int, running: float,
                          crossing: float, hybrid: bool)
        """
        super().__init__()
        env_utils.init_pygame()

        self.timeout = timeout
        self.client = env_utils.get_client(address, port, self.timeout)
        self.world: carla.World = self.client.get_world()
        self.synchronous_context = None
        self.sync_mode_enabled = False
        self.num_frames_to_skip = skip_frames

        # Time
        self.initial_timestamp: carla.Timestamp = None
        self.current_timestamp: carla.Timestamp = None

        # set fix fps:
        self.world_settings = carla.WorldSettings(no_rendering_mode=False,
                                                  synchronous_mode=False,
                                                  fixed_delta_seconds=1.0 / fps)
        self.world.apply_settings(self.world_settings)

        # Law compliance
        self.ignore_traffic_light = ignore_traffic_light

        # Map
        self.current_town = 'Town03'  # loaded by default

        if isinstance(town, str):
            self.set_town(town)

        self.map: carla.Map = self.world.get_map()

        # Vehicle
        self.vehicle_filter = vehicle_filter
        self.vehicle: carla.Vehicle = None
        self.control = carla.VehicleControl()

        # Weather
        self.weather = weather
        self.set_weather(weather)

        # Spawning (vehicles and pedestrians)
        self.vehicles = []
        self.walkers_ids = []
        self.should_spawn = isinstance(spawn, dict)
        self.spawn_dict = spawn

        # Path: origin, destination, and path-length:
        self.origin_type = 'map'  # 'map' means sample a random point from the world's map
        self.origin = None
        self.destination_type = 'map'
        self.destination = None
        self.path_length = None
        self.use_planner = True
        self.sampling_resolution = 2.0

        if isinstance(path, dict):
            origin_spec = path.get('origin', None)
            destination_spec = path.get('destination', None)
            self.path_length = path.get('length', None)

            # Origin:
            if isinstance(origin_spec, carla.Transform):
                self.origin = origin_spec
                self.origin_type = 'fixed'

            elif isinstance(origin_spec, dict):
                if 'point' in origin_spec:
                    self.origin = origin_spec['point']
                    self.origin_type = origin_spec.get('type', 'fixed')

                    assert isinstance(self.origin, carla.Transform)
                    assert self.origin_type in ['map', 'fixed', 'route']

                elif 'points' in origin_spec:
                    self.origins = origin_spec['points']
                    self.origin = None
                    self.origin_index = -1
                    self.origin_type = origin_spec.get('type', 'random')

                    assert isinstance(self.origins, list) and len(self.origins) > 0
                    assert all(isinstance(x, carla.Transform) for x in self.origins)
                    assert self.origin_type in ['random', 'sequential']

            elif isinstance(origin_spec, list):
                self.origins = origin_spec
                self.origin = None
                self.origin_index = -1
                self.origin_type = 'random'

            # Destination:
            if isinstance(destination_spec, carla.Location):
                self.destination = destination_spec
                self.destination_type = 'fixed'

            elif isinstance(destination_spec, dict):
                if 'point' in destination_spec:
                    self.destination = destination_spec['point']
                    self.destination_type = destination_spec.get('type', 'fixed')

                    assert isinstance(self.destination, carla.Location)
                    assert self.destination_type in ['map', 'fixed']

                elif 'points' in destination_spec:
                    self.destinations = destination_spec['points']
                    self.destination = None
                    self.destination_index = -1
                    self.destination_type = destination_spec.get('type', 'random')

                    assert isinstance(self.destinations, list) and len(self.destinations) > 0
                    assert all(isinstance(x, carla.Location) for x in self.destinations)
                    assert self.destination_type in ['random', 'sequential']

            # Path stuff:
            self.path_length = path.get('length', None)
            self.use_planner = path.get('use_planner', True)
            self.sampling_resolution = path.get('sampling_resolution', 2.0)

            if self.origin_type == 'route':
                assert self.destination_type == 'fixed'
                assert self.use_planner is True

        elif path is not None:
            raise ValueError('Argument [path] must be either "None" or a "dict".')

        # Path-planning:
        if self.use_planner:
            self.route = Route(planner=RoutePlanner(map=self.map, sampling_resolution=self.sampling_resolution))
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
            self.render_data = None  # some sensor_data to be rendered in render()
            self.window_size = window_size
            self.font = env_utils.get_font(size=13)
            self.display = env_utils.get_display(window_size)

        # vehicle sensors suite
        self.sensors = dict()

        # events and callbacks
        self.events: Dict[CARLAEvent, List[Callable]] = dict()

    @property
    def observation_space(self) -> spaces.Space:
        raise NotImplementedError

    @property
    def action_space(self) -> spaces.Space:
        raise NotImplementedError

    @property
    def info_space(self) -> spaces.Space:
        raise NotImplementedError

    @property
    def reward_range(self) -> tuple:
        raise NotImplementedError

    def reset(self) -> dict:
        print('env.reset')
        self.reset_world()
        self.trigger_event(event=CARLAEvent.RESET)

        if not self.sync_mode_enabled:
            self.__enter__()

        self.control = carla.VehicleControl()
        self.skip(num_frames=self.num_frames_to_skip)

        if self.should_spawn:
            self.spawn_actors(self.spawn_dict)
            self.spawn_dict = None
            self.should_spawn = False

        return self.get_observation(sensors_data={})

    def reward(self, actions, **kwargs):
        """Agent's reward function"""
        raise NotImplementedError

    def set_weather(self, weather: Union[carla.WeatherParameters, List[carla.WeatherParameters]]):
        """Sets the given weather. If [weather] is a list, a random preset from the list is chosen and set"""
        if isinstance(weather, list):
            weather = random.choice(weather)

        self.world.set_weather(weather)
        self.weather = weather
        print(f'Weather changed to {weather}.')

    def set_town(self, town: str, timeout=2.0, max_trials=5):
        """Loads then given town"""
        if self.current_town == town:
            print(f'{town} already loaded.')
            return

        print(f'Loading town: {town}...')
        self.map = None

        for _ in range(max_trials):
            try:
                self.world = self.client.load_world(town)
                self.map = self.world.get_map()
            except RuntimeError:
                print('Failed to connect to newly created map, retrying...')
                time.sleep(timeout)

            if self.map is not None:
                break

        self.world.apply_settings(self.world_settings)
        self.current_town = town
        print(f'Town {town} loaded.')

    def spawn_actors(self, spawn_dict: dict, hybrid=True, safe=True):
        """Instantiate vehicles and pedestrians in the current world"""
        traffic_manager = self.client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

        if spawn_dict.get('hybrid', hybrid):
            traffic_manager.set_hybrid_physics_mode(True)

        blueprints = env_utils.get_blueprints(self.world, vehicles_filter=spawn_dict.get('vehicles_filter', None),
                                              pedestrians_filter=spawn_dict.get('pedestrians_filter', None),
                                              safe=safe)
        # Spawn stuff
        self.vehicles = env_utils.spawn_vehicles(amount=spawn_dict.get('vehicles', 0), blueprints=blueprints[0],
                                                 client=self.client, spawn_points=self.map.get_spawn_points())

        self.walkers_ids = env_utils.spawn_pedestrians(amount=spawn_dict.get('pedestrians', 0),
                                                       blueprints=blueprints[1], client=self.client,
                                                       running=spawn_dict.get('running', 0.0),
                                                       crossing=spawn_dict.get('crossing', 0.0))

        traffic_manager.global_percentage_speed_difference(30.0)

    def destroy_actors(self):
        """Removes the previously spawned actors (vehicles and pedestrians/walkers)"""
        # Remove vehicles
        print(f'Destroying {len(self.vehicles)} vehicles...')
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles])

        # Stop walker controllers only (list is [controller, actor, controller, actor ...])
        actors = self.world.get_actors(self.walkers_ids)

        for i in range(0, len(self.walkers_ids), 2):
            actors[i].stop()

        print(f'Destroying {len(self.walkers_ids) // 2} pedestrians...')
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walkers_ids])

        time.sleep(1.0)

    @staticmethod
    def consume_pygame_events():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    return True

        return False

    def step(self, actions):
        pygame.event.get()
        self.clock.tick()

        sensors_data = self.world_step(actions)

        reward = self.reward(actions)
        terminal = self.terminal_condition()
        next_state = self.get_observation(sensors_data)

        return next_state, reward, terminal, self.get_info()

    def terminal_condition(self, **kwargs) -> Union[bool, int]:
        """Tells whether the episode is terminated or not."""
        raise NotImplementedError

    def close(self):
        print('env.close')
        super().close()

        self.destroy_actors()
        self.vehicles = []
        self.walkers_ids = []

        if self.vehicle:
            self.vehicle.destroy()

        if self.sync_mode_enabled:
            self.__exit__()

        for sensor in self.sensors.values():
            sensor.destroy()

    def define_sensors(self) -> dict:
        """Define which sensors should be equipped to the vehicle"""
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

    def unregister_event(self, event: CARLAEvent, callback):
        """Unregisters a given [callback] to a specific [event]"""
        assert isinstance(event, CARLAEvent)
        assert callable(callback)

        if event in self.events:
            callbacks = self.events[event]
            callbacks.remove(callback)
            self.events[event] = callbacks
        else:
            print(f'Event {event} not yet registered!')

    def render(self, mode='human'):
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
        raise NotImplementedError("Implement only if needed for pre-training.")

    def before_world_step(self):
        """Callback: called before world.tick()"""
        if self.ignore_traffic_light and self.vehicle.is_at_traffic_light():
            traffic_light = self.vehicle.get_traffic_light()
            traffic_light.set_state(carla.TrafficLightState.Green)

    def after_world_step(self, sensors_data: dict):
        """Callback: called after world.tick()."""
        self.current_timestamp = sensors_data['world'].timestamp

        if self.initial_timestamp is None:
            self.initial_timestamp = self.current_timestamp

    @staticmethod
    def on_sensors_data(data: dict) -> dict:
        """Callback. Triggers when a world's 'tick' occurs, meaning that data from sensors are been collected because a
        simulation step of the CARLA's world has been completed.
            - Use this method to preprocess sensors' output data for: rendering, observation, ...
        """
        return data

    def __enter__(self):
        """Enables synchronous mode.
           Usage:
              with carla_env as env:
                 # code...
        """
        self.synchronous_context.__enter__()
        self.sync_mode_enabled = True
        return self

    def __exit__(self, *args):
        """Disables synchronous mode"""
        self.synchronous_context.__exit__()
        self.sync_mode_enabled = False

        # propagate exception
        return False

    def world_step(self, actions):
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
            self.render_data = data
            self.render()
            self.render_data = None

            if self.should_debug:
                self.debug(actions)

            pygame.display.flip()

        return data

    def reset_world(self):
        # choose origin (spawn point)
        if self.origin_type == 'map':
            self.origin = env_utils.random_spawn_point(self.map)

        elif self.origin_type == 'random':
            self.origin = random.choice(self.origins)

        elif self.origin_type == 'sequential':
            self.origin_index = (self.origin_index + 1) % len(self.origins)
            self.origin = self.origins[self.origin_index]

        # choose destination (final point)
        if self.destination_type == 'map':
            self.destination = env_utils.random_spawn_point(self.map, different_from=self.origin.location).location

        elif self.destination_type == 'random':
            self.destination = random.choice(self.destinations)  # TODO: ensure different from origin?

        elif self.destination_type == 'sequential':
            self.destination_index = (self.destination_index + 1) % len(self.destinations)
            self.destination = self.destinations[self.destination_index]

        # plan path between origin and destination
        if self.use_planner:
            self.route.plan(origin=self.origin.location, destination=self.destination)

        # spawn actor
        if self.vehicle is None:
            blueprint = env_utils.random_blueprint(self.world, actor_filter=self.vehicle_filter)
            self.vehicle: carla.Vehicle = env_utils.spawn_actor(self.world, blueprint, self.origin)

            self._create_sensors()
            self.synchronous_context = CARLASyncContext(self.world, self.sensors, fps=self.fps)
        else:
            self.vehicle.apply_control(carla.VehicleControl())
            self.vehicle.set_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))

            if self.origin_type == 'route':
                new_origin = self.route.random_waypoint().transform
                self.vehicle.set_transform(new_origin)
            else:
                self.vehicle.set_transform(self.origin)

    def actions_to_control(self, actions):
        """Specifies the mapping between an actions vector and the vehicle's control."""
        raise NotImplementedError

    def get_observation(self, sensors_data: dict) -> dict:
        raise NotImplementedError

    def get_info(self) -> dict:
        return {}

    def elapsed_time(self):
        """Returns the total elapsed time in seconds, computed from the last reset() call."""
        return self.current_timestamp.elapsed_seconds - self.initial_timestamp.elapsed_seconds

    def available_towns(self) -> list:
        """Returns a list with the names of the currently available maps/towns"""
        return list(map(lambda s: s.split('/')[-1], self.client.get_available_maps()))

    def _create_sensors(self):
        for name, args in self.define_sensors().items():
            if args is None:
                continue

            kwargs = args.copy()
            sensor = Sensor.create(sensor_type=kwargs.pop('type'), parent_actor=self.vehicle, **kwargs)

            if name == 'world':
                raise ValueError(f'Cannot name a sensor `world` because is reserved.')

            self.sensors[name] = sensor


# TODO: make wrappers be composable? (e.g. treat them as environments)
class CARLAWrapper(gym.Wrapper):
    pass


class CARLAPlayWrapper(CARLAWrapper):
    """Makes an already instantiated CARLAEnvironment be playable with a keyboard"""
    CONTROL = dict(type='float', shape=(5,), min_value=-1.0, max_value=1.0,
                   default=[0.0, 0.0, 0.0, 0.0, 0.0])

    def __init__(self, env: CARLABaseEnvironment):
        super().__init__(env)
        print('Controls: (W, or UP) accelerate, (A or LEFT) steer left, (D or RIGHT) steer right, (S or DOWN) brake, '
              '(Q) toggle reverse, (SPACE) hand-brake, (ESC) quit.')
        self.env = env
        self._steer_cache = 0.0

        # Wrap environment's methods:
        self.env.actions_to_control = lambda actions: self.actions_to_control(self.env, actions)
        self.env.before_world_step = lambda: self.before_world_step(self.env)

    def reset(self) -> dict:
        self._steer_cache = 0.0
        return self.env.reset()

    def play(self):
        """Let's you control the vehicle with a keyboard."""
        state = self.reset()
        done = False

        try:
            with self.env.synchronous_context:
                while not done:
                    actions = self.get_action(state)
                    state, reward, done, info = self.env.step(actions)
        finally:
            self.env.close()

    def get_action(self, state):
        return self._parse_events()

    def _parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()

            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    raise Exception('closing...')

                elif event.key == K_q:
                    self.env.control.gear = 1 if self.env.control.reverse else -1

        return self._parse_vehicle_keys()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

    def _parse_vehicle_keys(self):
        keys = pygame.key.get_pressed()
        steer_increment = 5e-4 * self.env.clock.get_time()

        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment

        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0

        self._steer_cache = min(1.0, max(-1.0, self._steer_cache))
        self.env.control.reverse = self.env.control.gear < 0

        # actions
        throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer = round(self._steer_cache, 1)
        brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        reverse = 1.0 if self.env.control.reverse else 0.0
        hand_brake = keys[K_SPACE]

        return [throttle, steer, brake, reverse, hand_brake]

    @staticmethod
    def actions_to_control(env, actions):
        env.control.throttle = actions[0]
        env.control.steer = actions[1]
        env.control.brake = actions[2]
        env.control.reverse = bool(actions[3])
        env.control.hand_brake = bool(actions[4])

    @staticmethod
    def before_world_step(env):
        if env.should_debug:
            env.route.draw_route(env.world.debug, life_time=1.0 / env.fps)
            # env.route.draw_next_waypoint(env.world.debug, env.vehicle.get_location(), life_time=1.0 / env.fps)


class CARLACollectWrapper(CARLAWrapper):
    """Wraps a CARLA Environment, collecting input observations and output actions that can be later
       used for pre-training or imitation learning purposes.
    """

    def __init__(self, env: CARLABaseEnvironment, ignore_traffic_light: bool, traces_dir='traces', name='carla',
                 behaviour='normal'):
        super().__init__(env)
        self.env = env
        self.agent = None
        self.agent_behaviour = behaviour  # 'normal', 'cautious', or 'aggressive'
        self.ignore_traffic_light = ignore_traffic_light

        # Saving & Buffers
        self.save_dir = rl_utils.makedir(traces_dir, name)
        print('save_dir:', self.save_dir)
        self.buffer = None
        self.timestep = 0

        # Check for collisions
        self.have_collided = False
        self.should_terminate = False

    def reset(self) -> dict:
        self.timestep = 0
        observation = self.env.reset()

        self.agent = BehaviorAgent(vehicle=self.env.vehicle, behavior=self.agent_behaviour,
                                   ignore_traffic_light=self.ignore_traffic_light)
        self.agent.set_destination(start_location=self.env.vehicle.get_location(), end_location=self.env.destination,
                                   clean=True)
        return observation

    def on_collision(self, actor):
        self.have_collided = True

        if 'pedestrian' in actor:
            self.should_terminate = True
        elif 'vehicle' in actor:
            self.should_terminate = True
        else:
            self.should_terminate = False

    def collect(self, episodes: int, timesteps: int, agent_debug=False, episode_reward_threshold=0.0, close=True):
        self.init_buffer(num_timesteps=timesteps)
        env = self.env
        env.register_event(event=CARLAEvent.ON_COLLISION, callback=self.on_collision)

        self.have_collided = False
        self.should_terminate = False

        try:
            for episode in range(episodes):
                state = self.reset()
                episode_reward = 0.0

                for t in range(timesteps):
                    # act
                    self.agent.update_information(vehicle=self.vehicle)
                    control = self.agent.run_step(debug=agent_debug)
                    action = env.control_to_actions(control)

                    # step
                    next_state, reward, done, info = env.step(action)
                    episode_reward += reward

                    if self.have_collided:
                        self.have_collided = False

                        if self.should_terminate:
                            episode_reward = -episode_reward_threshold
                            break

                    # record
                    self.store_transition(state=state, action=action, reward=reward, done=done, info=info)
                    state = next_state

                    if done or (t == timesteps - 1):
                        buffer = self.end_trajectory()
                        break

                if episode_reward > episode_reward_threshold:
                    self.serialize(buffer, episode)
                    print(f'Trace-{episode} saved with reward={round(episode_reward, 2)}.')
                else:
                    print(f'Trace-{episode} discarded because reward={round(episode_reward, 2)} below threshold!')
        finally:
            env.unregister_event(event=CARLAEvent.ON_COLLISION, callback=self.on_collision)
            if close:
                env.close()

    def init_buffer(self, num_timesteps: int):
        # partial buffer: misses 'state', 'action', and 'info'
        self.buffer = dict(reward=np.zeros(shape=num_timesteps),
                           done=np.zeros(shape=num_timesteps))

        obs_spec = rl_utils.space_to_spec(space=self.env.observation_space)
        act_spec = rl_utils.space_to_spec(space=self.env.action_space)
        info_spec = rl_utils.space_to_spec(space=self.env.info_space)

        print('obs_spec\n', obs_spec)
        print('action_spec\n', act_spec)
        print('info_spec\n', info_spec)

        # include in buffer 'state' and 'action'
        self._apply_space_spec(spec=obs_spec, size=num_timesteps, name='state')
        self._apply_space_spec(spec=act_spec, size=num_timesteps, name='action')
        self._apply_space_spec(spec=info_spec, size=num_timesteps, name='info')
        self.timestep = 0

    def store_transition(self, **kwargs):
        """Collects one transition (s, a, r, d, i)"""
        for name, value in kwargs.items():
            self._store_item(item=value, index=self.timestep, name=name)

        self.timestep += 1

    def end_trajectory(self) -> dict:
        """Ends a sequence of transitions {(s, a, r, d, i)_t}"""
        # Add the reward for the terminal/final state:
        self.buffer['reward'] = np.concatenate([self.buffer['reward'], np.array([0.0])])

        # Duplicate the buffer and cut off the exceeding part (if any)
        buffer = dict()

        for key, value in self.buffer.items():
            buffer[key] = value[:self.timestep]

        buffer['reward'] = self.buffer['reward'][:self.timestep + 1]
        return buffer

    def serialize(self, buffer: dict, episode: int):
        """Writes to file (npz - numpy compressed format) all the transitions collected so far"""
        # Trace's file path:
        filename = f'trace-{episode}-{time.strftime("%Y%m%d-%H%M%S")}.npz'
        trace_path = os.path.join(self.save_dir, filename)

        # Save buffer
        np.savez_compressed(file=trace_path, **buffer)
        print(f'Trace {filename} saved.')

    def _apply_space_spec(self, spec: Union[tuple, dict], size: int, name: str):
        if not isinstance(spec, dict):
            shape = (size,) + spec
            self.buffer[name] = np.zeros(shape=shape, dtype=np.float32)
            return

        # use recursion + names to handle arbitrary nested dicts and recognize them
        for spec_name, sub_spec in spec.items():
            self._apply_space_spec(spec=sub_spec, size=size, name=f'{name}_{spec_name}')

    def _store_item(self, item, index: int, name: str):
        if not isinstance(item, dict):
            self.buffer[name][index] = item
            return

        # recursion
        for key, value in item.items():
            self._store_item(item=value, index=index, name=f'{name}_{key}')


class CARLARecordWrapper:
    """Wraps a CARLA Environment in order to record input observations"""
    pass


# -------------------------------------------------------------------------------------------------
# -- Implemented CARLA Environments
# -------------------------------------------------------------------------------------------------

class OneCameraCARLAEnvironment(CARLABaseEnvironment):
    """One camera (front) CARLA Environment"""
    # Control: throttle or brake, steer, reverse
    ACTION = dict(space=spaces.Box(low=-1.0, high=1.0, shape=(3,)), default=np.zeros(shape=3, dtype=np.float32))
    CONTROL = dict(space=spaces.Box(low=-1.0, high=1.0, shape=(4,)), default=np.zeros(shape=4, dtype=np.float32))

    # Vehicle: speed, acceleration, angular velocity, similarity, distance to waypoint
    VEHICLE_FEATURES = dict(space=spaces.Box(low=np.array([0.0, -np.inf, 0.0, -1.0, -np.inf]),
                                             high=np.array([15.0, np.inf, np.inf, 1.0, np.inf])),
                            default=np.zeros(shape=5, dtype=np.float32))

    # Road: intersection (bool), junction (bool), speed_limit, traffic_light (presence + state), lane type and change,
    ROAD_FEATURES = dict(space=spaces.Box(low=np.zeros(shape=(9,)),
                                          high=np.array([1.0, 1.0, 15.0, 1.0, 4.0, 2.0, 10.0, 10.0, 3.0])),
                         default=np.zeros(shape=9, dtype=np.float32))

    # High-level routing command (aka RoadOption)
    COMMAND_SPACE = spaces.Box(low=0.0, high=1.0, shape=RoadOption.VOID.shape)

    INFO_SPACE = spaces.Dict(speed=spaces.Box(low=0.0, high=150.0, shape=(1,)),
                             speed_limit=spaces.Box(low=0.0, high=90.0, shape=(1,)),
                             similarity=spaces.Box(low=-1.0, high=1.0, shape=(1,)),
                             distance_to_next_waypoint=spaces.Box(low=0.0, high=np.inf, shape=(1,)))

    def __init__(self, *args, disable_reverse=False, min_throttle=0.0, camera='segmentation',
                 hard_control_threshold: Union[float, None] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_space = spaces.Box(low=0.0, high=1.0, shape=self.image_shape)
        self.camera_type = camera

        # control hack
        self.disable_reverse = disable_reverse
        self.min_throttle = min_throttle
        self.should_harden_controls = isinstance(hard_control_threshold, float)
        self.hard_control_threshold = hard_control_threshold

        # reward computation
        self.collision_penalty = 0.0
        self.should_terminate = False
        self.similarity = 0.0
        self.forward_vector = None

        self.next_command = RoadOption.VOID
        self.last_actions = self.ACTION['default']
        self.last_location = None
        self.last_travelled_distance = 0.0
        self.total_travelled_distance = 0.0

        # Observations
        self.default_image = np.zeros(shape=self.image_shape, dtype=np.float32)

    @property
    def action_space(self) -> spaces.Space:
        return self.ACTION['space']

    @property
    def observation_space(self) -> spaces.Space:
        return spaces.Dict(road=self.ROAD_FEATURES['space'], vehicle=self.VEHICLE_FEATURES['space'],
                           past_control=self.CONTROL['space'], command=self.COMMAND_SPACE, image=self.image_space)

    @property
    def info_space(self) -> spaces.Space:
        return self.INFO_SPACE

    @property
    def reward_range(self) -> tuple:
        return -float('inf'), float('inf')

    def reward(self, actions, time_cost=-1, d=2.0, w=3.0, s=2.0, v_max=150.0, d_max=100.0, **kwargs) -> float:
        # Direction term: alignment of the vehicle's forward vector with the waypoint's forward vector
        speed = min(utils.speed(self.vehicle), v_max)

        if 0.75 <= self.similarity <= 1.0:
            direction_penalty = speed * self.similarity
        else:
            direction_penalty = (speed + 1.0) * abs(self.similarity) * -d
            self.trigger_event(CARLAEvent.OUT_OF_LANE, similarity=self.similarity)

        # Distance from waypoint (and also lane center)
        waypoint_term = min(self.route.distance_to_next_waypoint(), d_max)
        waypoint_term = -waypoint_term if waypoint_term <= 5.0 else waypoint_term * -w

        # Speed-limit compliance:
        speed_limit = self.vehicle.get_speed_limit()
        speed_penalty = s * (speed_limit - speed) if speed > speed_limit else 0.0

        return time_cost - self.collision_penalty + waypoint_term + direction_penalty + speed_penalty

    def step(self, actions):
        state, reward, done, info = super().step(actions)
        self.collision_penalty = 0.0
        self.last_travelled_distance = 0.0

        return state, reward, done, info

    def reset(self) -> dict:
        self.last_actions = self.ACTION['default']
        self.should_terminate = False
        self.total_travelled_distance = 0.0
        self.last_travelled_distance = 0.0
        self.next_command = RoadOption.VOID

        # reset observations:
        observation = super().reset()

        self.last_location = self.vehicle.get_location()
        # self.last_location = self.origin.location
        return observation

    def terminal_condition(self, **kwargs) -> bool:
        if self.should_terminate:
            return True

        return self.route.distance_to_destination(self.vehicle.get_location()) <= 2.0

    def define_sensors(self) -> dict:
        if self.camera_type == 'rgb':
            camera_sensor = SensorSpecs.rgb_camera(position='on-top2', attachment_type='Rigid',
                                                   image_size_x=self.image_size[0],
                                                   image_size_y=self.image_size[1],
                                                   sensor_tick=self.tick_time)
            depth_sensor = None
        else:
            camera_sensor = SensorSpecs.segmentation_camera(position='on-top2', attachment_type='Rigid',
                                                            image_size_x=self.image_size[0],
                                                            image_size_y=self.image_size[1],
                                                            sensor_tick=self.tick_time)

            depth_sensor = SensorSpecs.depth_camera(position='on-top2', attachment_type='Rigid',
                                                    image_size_x=self.image_size[0],
                                                    image_size_y=self.image_size[1],
                                                    sensor_tick=self.tick_time)

        return dict(collision=SensorSpecs.collision_detector(callback=self.on_collision),
                    imu=SensorSpecs.imu(),
                    camera=camera_sensor,
                    depth=depth_sensor)

    def on_collision(self, event: carla.CollisionEvent, penalty=1000.0):
        actor_type = event.other_actor.type_id
        print(f'Collision with actor={actor_type})')
        self.trigger_event(event=CARLAEvent.ON_COLLISION, actor=actor_type)

        if 'pedestrian' in actor_type:
            self.collision_penalty += penalty
            self.should_terminate = True

        elif 'vehicle' in actor_type:
            self.collision_penalty += penalty / 2.0
            self.should_terminate = True
        else:
            self.collision_penalty += penalty / 100.0
            self.should_terminate = False

    def render(self, mode='human'):
        assert self.render_data is not None
        image = self.render_data['camera']
        env_utils.display_image(self.display, image, window_size=self.window_size)

    def debug_text(self, actions):
        speed_limit = self.vehicle.get_speed_limit()
        speed = utils.speed(self.vehicle)
        distance = self.total_travelled_distance

        if speed > speed_limit:
            speed_text = dict(text='Speed %.1f km/h' % speed, color=(255, 0, 0))
        else:
            speed_text = 'Speed %.1f km/h' % speed

        if self.similarity >= 0.75:
            similarity_text = 'Similarity %.2f' % self.similarity
        else:
            similarity_text = dict(text='Similarity %.2f' % self.similarity, color=(255, 0, 0))

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
                similarity_text,
                'Waypoint\'s Distance %.2f' % self.route.distance_to_next_waypoint(),
                'Route Option: %s' % self.next_command.name,
                'OP: %s' % self.next_command.to_one_hot(),
                '',
                'Reward: %.2f' % self.reward(actions),
                'Collision penalty: %.2f' % self.collision_penalty]

    def control_to_actions(self, control: carla.VehicleControl):
        reverse = bool(control.reverse)

        if control.throttle > 0.0:
            return [control.throttle, control.steer, reverse]

        return [-control.brake, control.steer, reverse]

    def on_sensors_data(self, data: dict) -> dict:
        data['camera'] = self.sensors['camera'].convert_image(data['camera'])

        if 'depth' in self.sensors:
            # include depth information in one image:
            data['depth'] = self.sensors['depth'].convert_image(data['depth'])
            data['camera'] = np.multiply(1 - data['depth'] / 255.0, data['camera'])

        if self.image_shape[2] == 1:
            data['camera'] = env_utils.cv2_grayscale(data['camera'])

        return data

    def after_world_step(self, sensors_data: dict):
        super().after_world_step(sensors_data)
        self._update_env_state()

    def actions_to_control(self, actions):
        self.control.throttle = max(self.min_throttle, float(actions[0]) if actions[0] > 0 else 0.0)
        # self.control.throttle = float(actions[0]) if actions[0] > 0 else 0.0
        self.control.brake = float(-actions[0]) if actions[0] < 0 else 0.0
        self.control.steer = float(actions[1])
        self.control.hand_brake = False

        if self.should_harden_controls and (utils.speed(self.vehicle) <= self.hard_control_threshold):
            self.control.throttle = float(round(self.control.throttle))
            self.control.brake = float(round(self.control.brake))

        if self.disable_reverse:
            self.control.reverse = False
        else:
            self.control.reverse = bool(actions[2] > 0)

    def get_observation(self, sensors_data: dict) -> dict:
        if len(sensors_data.keys()) == 0:
            # sensor_data is empty so, return a default observation
            return dict(image=self.default_image, vehicle=self.VEHICLE_FEATURES['default'],
                        road=self.ROAD_FEATURES['default'], past_control=self.CONTROL['default'],
                        command=RoadOption.VOID.to_one_hot())

        image = sensors_data['camera']

        # resize image if necessary
        if image.shape != self.image_shape:
            image = env_utils.resize(image, size=self.image_size)

        # 0-1 scaling
        image /= 255.0

        # observations
        vehicle_obs = self._get_vehicle_features()
        control_obs = self._control_as_vector()
        road_obs = self._get_road_features()

        obs = dict(image=image, vehicle=vehicle_obs, road=road_obs, past_control=control_obs,
                   command=self.next_command.to_one_hot())

        return env_utils.replace_nans(obs)

    def get_info(self) -> dict:
        """Returns a dict with additional information either for debugging or learning"""
        return dict(speed=utils.speed(self.vehicle), speed_limit=self.vehicle.get_speed_limit(),
                    similarity=self.similarity, distance_to_next_waypoint=self.route.distance_to_next_waypoint())

    def _control_as_vector(self) -> list:
        return [self.control.throttle, self.control.brake, self.control.steer, float(self.control.reverse)]

    def _get_road_features(self):
        waypoint: carla.Waypoint = self.map.get_waypoint(self.vehicle.get_location())
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
            lane_type = 2  # other

        return np.array([waypoint.is_intersection,
                         waypoint.is_junction,
                         round(speed_limit / 10.0),
                         # Traffic light:
                         is_at_traffic_light,
                         WAYPOINT_DICT['traffic_light'][traffic_light_state],
                         # Lanes:
                         lane_type,
                         WAYPOINT_DICT['lane_marking_type'][waypoint.left_lane_marking.type],
                         WAYPOINT_DICT['lane_marking_type'][waypoint.right_lane_marking.type],
                         WAYPOINT_DICT['lane_change'][waypoint.lane_change]], dtype=np.float32)

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
            self.next_command = self.route.next.road_op

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
        self.total_travelled_distance += abs(self.last_travelled_distance)
        self.last_location = location2


class OneCameraCARLAEnvironmentDiscrete(OneCameraCARLAEnvironment):
    """One-camera CARLA Environment with discrete action-space"""

    def __init__(self, bins: int, *args, **kwargs):
        assert (bins >= 2) and (bins % 2 == 0)

        action_space = self.ACTION['space']
        assert isinstance(action_space, spaces.Box)

        self.bins = bins
        self._low = action_space.low
        self._delta = (action_space.high - action_space.low) / bins

        # change action space to "discrete"
        self.ACTION = dict(space=spaces.MultiDiscrete([self.bins] * 3),
                           default=np.zeros(shape=3, dtype=np.float32))

        super().__init__(*args, **kwargs)

    def actions_to_control(self, actions):
        super().actions_to_control(actions=self.to_continuous(actions))

    def to_continuous(self, discrete_actions: list):
        """Maps a discrete array of bins into their corresponding continuous values"""
        return self._delta * np.asarray(discrete_actions) + self._low

    def control_to_actions(self, control: carla.VehicleControl):
        actions = super().control_to_actions(control)
        return self.to_discrete(actions)

    def to_discrete(self, continuous_actions: list):
        """Maps a continuous array of values into their corresponding bins (i.e. inverse of `interpolate`)"""
        return ((np.asarray(continuous_actions) - self._low) / self._delta).astype('int')


class ThreeCameraCARLAEnvironment(OneCameraCARLAEnvironment):
    """Three Camera (front, lateral left and right) CARLA Environment"""

    def __init__(self, *args, image_shape=(120, 160, 1), window_size=(600, 300), **kwargs):
        # Make the shape of the final image three times larger to account for the three cameras
        image_shape = (image_shape[0], image_shape[1] * 3, image_shape[2])

        super().__init__(*args, image_shape=image_shape, window_size=window_size, **kwargs)
        self.image_size = (image_shape[1] // 3, image_shape[0])

    def define_sensors(self) -> dict:
        return dict(collision=SensorSpecs.collision_detector(callback=self.on_collision),
                    imu=SensorSpecs.imu(),
                    front_camera=SensorSpecs.segmentation_camera(position='on-top2', attachment_type='Rigid',
                                                                 image_size_x=self.image_size[0],
                                                                 image_size_y=self.image_size[1],
                                                                 sensor_tick=self.tick_time),
                    depth=SensorSpecs.depth_camera(position='on-top2', attachment_type='Rigid',
                                                   image_size_x=self.image_size[0],
                                                   image_size_y=self.image_size[1],
                                                   sensor_tick=self.tick_time),
                    left_camera=SensorSpecs.segmentation_camera(position='lateral-left', attachment_type='Rigid',
                                                                image_size_x=self.image_size[0],
                                                                image_size_y=self.image_size[1],
                                                                sensor_tick=self.tick_time),
                    right_camera=SensorSpecs.segmentation_camera(position='lateral-right', attachment_type='Rigid',
                                                                 image_size_x=self.image_size[0],
                                                                 image_size_y=self.image_size[1],
                                                                 sensor_tick=self.tick_time))

    def render(self, mode='human'):
        assert self.render_data is not None
        image = self.render_data['camera']
        env_utils.display_image(self.display, image, window_size=self.window_size)

    def on_sensors_data(self, data: dict) -> dict:
        front_image = self.sensors['front_camera'].convert_image(data['front_camera'])
        left_image = self.sensors['left_camera'].convert_image(data['left_camera'])
        right_image = self.sensors['right_camera'].convert_image(data['right_camera'])

        # include depth information in one image:
        if 'depth' in self.sensors:
            data['depth'] = self.sensors['depth'].convert_image(data['depth'])
            front_image = np.multiply(1 - data['depth'] / 255.0, front_image)

        # Concat images
        data['camera'] = np.concatenate((left_image, front_image, right_image), axis=1)

        if self.image_shape[2] == 1:
            data['camera'] = env_utils.cv2_grayscale(data['camera'])

        return data


class ThreeCameraCARLAEnvironmentDiscrete(ThreeCameraCARLAEnvironment):
    """Three-camera CARLA Environment with discrete action-space"""

    def __init__(self, bins: int, *args, **kwargs):
        assert (bins >= 2) and (bins % 2 == 0)

        action_space = self.ACTION['space']
        assert isinstance(action_space, spaces.Box)

        self.bins = bins
        self._low = action_space.low
        self._delta = (action_space.high - action_space.low) / bins

        # change action space to "discrete"
        self.ACTION = dict(space=spaces.MultiDiscrete([self.bins] * 3),
                           default=np.zeros(shape=3, dtype=np.float32))

        super().__init__(*args, **kwargs)

    def actions_to_control(self, actions):
        print(f'actions_to_control: d{actions} -> c{self.to_continuous(actions)}')
        super().actions_to_control(actions=self.to_continuous(actions))

    def to_continuous(self, discrete_actions: list):
        """Maps a discrete array of bins into their corresponding continuous values"""
        return self._delta * np.asarray(discrete_actions) + self._low

    def control_to_actions(self, control: carla.VehicleControl):
        actions = super().control_to_actions(control)
        print(f'control_to_actions: {control} -> c{actions} -> d{self.to_discrete(actions)}')
        return self.to_discrete(actions)

    def to_discrete(self, continuous_actions: list):
        """Maps a continuous array of values into their corresponding bins (i.e. inverse of `interpolate`)"""
        return ((np.asarray(continuous_actions) - self._low) / self._delta).astype('int')


# -------------------------------------------------------------------------------------------------
# -- Benchmarks: CARLA + NoCrash
# -------------------------------------------------------------------------------------------------

# TODO: untested
class CARLABenchmark(CARLAWrapper):
    """CARLA benchmark, as described in the paper: "End-to-end Driving via Conditional Imitation Learning"
         - https://arxiv.org/pdf/1710.02410

       The agent is evaluated on:
         - Town: "Town02".
         - Performance are measured in two ways: (1) success rate, and (2) avg. distance without infractions.
         - Six weather presets (almost like in "Controllable Imitative Reinforcement Learning" paper):
            1. CloudyNoon,
            2. SoftRainSunset,
            3. CloudyNoon,
            4. MidRainyNoon,
            5. CloudySunset,
            6. HardRainSunset.
         - An episode terminates when an "infraction". An infraction occurs when the agent is not able to
           reach the goal location within the time-budget, and/or when the agent drives in the opposite road
           segment (in this case, this kind of infraction is detected by measuring "direction-similarity" with
           the next correct waypoint).
         - Time-budget: in this case, the time budget is represented by "average speed".

       Details:
       - https://github.com/carla-simulator/driving-benchmarks
    """
    TRAIN_TOWN = 'Town01'
    TEST_TOWN = 'Town02'
    TRAIN_WEATHERS = [carla.WeatherParameters.ClearNoon,
                      carla.WeatherParameters.ClearSunset,
                      carla.WeatherParameters.SoftRainNoon,
                      carla.WeatherParameters.SoftRainSunset]
    TEST_WEATHERS = [carla.WeatherParameters.CloudyNoon,
                     carla.WeatherParameters.SoftRainSunset,
                     carla.WeatherParameters.WetCloudyNoon,
                     carla.WeatherParameters.MidRainyNoon,
                     carla.WeatherParameters.CloudySunset,
                     carla.WeatherParameters.HardRainSunset]

    class Tasks(enum.Enum):
        """Kind of tasks that the benchmark supports"""
        EMPTY_TOWN = 0
        REGULAR_TRAFFIC = 1
        DENSE_TRAFFIC = 2

    # Specifications of each task for training/testing evaluation:
    TASKS_SPEC = {Tasks.EMPTY_TOWN: {
        TRAIN_TOWN: dict(vehicles=0, pedestrians=0),
        TEST_TOWN: dict(vehicles=0, pedestrians=0)},

        Tasks.REGULAR_TRAFFIC: {
            TRAIN_TOWN: dict(vehicles=20, pedestrians=50),
            TEST_TOWN: dict(vehicles=15, pedestrians=50)},

        Tasks.DENSE_TRAFFIC: {
            TRAIN_TOWN: dict(vehicles=100, pedestrians=250),
            TEST_TOWN: dict(vehicles=70, pedestrians=150)}}

    def __init__(self, env: CARLABaseEnvironment, task: Tasks, preset='test', weather=None, avg_speed=10.0):
        assert isinstance(task, CARLABenchmark.Tasks)
        assert preset in ['test', 'train']
        super().__init__(env)

        self.env = env
        self.is_out_of_lane = False
        self.has_collided = False

        # metrics
        self.successful = []
        self.route_length = None
        self.time_limit = None
        self.avg_speed = avg_speed

        # events
        # self.env.register_event(CARLAEvent.OUT_OF_LANE, callback=self.on_out_of_lane)
        self.env.register_event(CARLAEvent.ON_COLLISION, callback=self.on_collision)

        # prepare stuff for benchmark
        if preset == 'test':
            self.task_spec = self.TASKS_SPEC[task][self.TEST_TOWN]
            self.env.set_town(self.TEST_TOWN)
        else:
            self.task_spec = self.TASKS_SPEC[task][self.TRAIN_TOWN]
            self.env.set_town(self.TRAIN_TOWN)

        if weather is None:
            weather = self.TEST_WEATHERS

        self.env.set_weather(weather)
        self.env.spawn_actors(spawn_dict=self.task_spec)

    def reset(self):
        self.env.reset()
        self.is_out_of_lane = False
        self.has_collided = False
        self.route_length = self.env.route.distance_to_destination(self.env.destination)
        self.time_limit = self.route_length / self.avg_speed * 3.6

    def on_collision(self, actor):
        if 'sidewalk' in actor:
            self.has_collided = False
        else:
            self.has_collided = True

    def on_out_of_lane(self, **kwargs):
        self.is_out_of_lane = True

    def destination_reached(self, threshold=2.0) -> bool:
        """Tells whether or not the agent has reached the goal (destination) location"""
        return self.env.route.distance_to_destination(self.env.vehicle.get_location()) <= threshold

    def step(self, actions):
        next_state, reward, terminal, info = self.env.step(actions)

        if self.env.elapsed_time() < self.time_limit:
            if terminal:
                self.successful.append(True)
        else:
            terminal = True
            self.successful.append(False)

        # benchmark's termination condition
        # terminal |= self.is_out_of_lane
        terminal |= self.has_collided

        if self.env.elapsed_time() > self.time_limit:
            terminal = True
            self.successful.append(False)
        elif terminal:
            self.successful.append(self.destination_reached())

        # reset flags
        self.is_out_of_lane = False
        self.has_collided = False

        return next_state, reward, terminal, info

    def success_rate(self) -> float:
        """Returns the success rate: num. of successful episodes"""
        if len(self.successful) == 0:
            return 0.0

        return sum(self.successful) / len(self.successful) * 100.0

    def close(self):
        self.env.close()
