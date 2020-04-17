"""Contains various environments build to work with CARLA simulator."""

import pygame

from typing import Optional
from datetime import datetime

from tensorforce import Environment, Agent

from agents import env_utils
from agents.sensors import *

from navigation import Route, RoutePlanner

from tools import misc, utils
from tools.utils import WAYPOINT_DICT, profile
from tools.synchronous_mode import CARLASyncContext

MAX_RADIANS = 2.0 * math.pi
MAX_SPEED = 150.0


# TODO: think about sensors' callbacks
class SynchronousCARLAEnvironment(Environment):
    """A TensorForce Environment for the CARLA driving simulator.
        - This environment is "synchronized" with the server, meaning that the server waits for a client tick.
        - Subclass to customize the behaviour of states, actions, sensors, reward function, agent, training loop, etc.

       Requires, you to:
        - Download and Install the CARLA simulator,
        - Run the CARLA simulator from command line.
    """
    # States and actions specifications:
    # Actions: throttle, steer, brake, reverse (bool), hand_brake (bool)
    ACTIONS_SPEC = dict(type='float', shape=(4,), min_value=-1.0, max_value=1.0)
    DEFAULT_ACTIONS = np.array([0.0, 0.0, 0.0, 0.0])

    # Vehicle: speed, gear, accelerometer (x, y, z), gyroscope (x, y, z), position (x, y), compass
    VEHICLE_FEATURES_SPEC = dict(type='float', shape=(11,))

    # Road: intersection (bool), junction (bool), speed_limit, traffic_light (presence + state), lane_width,
    #       lane_change, left_lane, right_lane
    ROAD_FEATURES_SPEC = dict(type='float', shape=(10,))

    # TODO: add a loading map functionality (specified or at random) - load_map
    # TODO: add should_debug flag(s)
    # TODO: could change 'route_resolution' to 'planner' in order to abstract the inner planner, also making it optional
    def __init__(self, address='localhost', port=2000, timeout=2.0, image_shape=(200, 150, 3), window_size=(800, 600),
                 vehicle_filter='vehicle.*', sensors: dict = None, route_resolution=2.0, fps=30.0, visualize=True,
                 debug=False):
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
        self.fps = fps
        self.visualize = visualize
        self.should_debug = debug
        self.clock = pygame.time.Clock()

        if self.visualize:
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

    def states(self):
        # TODO: when stacking feature vectors, reshape them into a 2D matrix so that convolutions can be applied!!
        # TODO: consider to include past (more than one) skills, but one-hot encoded!
        return dict(image=dict(shape=self.image_shape),
                    vehicle_features=self.VEHICLE_FEATURES_SPEC,
                    road_features=self.ROAD_FEATURES_SPEC,
                    previous_actions=self.ACTIONS_SPEC)

    def actions(self):
        return self.ACTIONS_SPEC

    def reset(self, soft=False):
        print('env.reset(soft=True)') if soft else print('env.reset')
        self._reset_world(soft=soft)

        # reset actions
        self.control = carla.VehicleControl()
        self.prev_actions = self.DEFAULT_ACTIONS

        return self._get_observation(image=None)

    def reward(self, actions, time_cost=-1.0, b=2.0, c=2.0, d=2.0):
        """Agent's reward function"""
        # TODO: include a penalty for law compliance: exceeding the speed limit, red traffic light...

        # Direction term: alignment of the vehicle's forward vector with the waypoint's forward vector
        speed = utils.speed(self.vehicle)
        similarity = self.similarity
        # self.similarity = similarity * (-1 if self.control.reverse else +1)

        if 0.8 <= similarity <= 1.0:
            direction_penalty = (speed + 1) * similarity  # speed + 1, to avoid 0 speed
        else:
            direction_penalty = (speed + 1) * abs(similarity) * -d

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

    @profile
    def execute(self, actions, record_path: str = None):
        self.prev_actions = actions

        # https://stackoverflow.com/questions/20165492/pygame-window-not-responding-after-a-few-seconds
        pygame.event.get()
        # pygame.event.pump()
        self.clock.tick()

        image = self._sync_world_step(actions, record_path=record_path)

        reward = self.reward(actions)
        terminal = self.terminal_condition()
        next_state = self._get_observation(image)

        # TODO: penalize remaining distance to destination when terminal=True?

        # Reset collision count
        self.collision_penalty = 0.0
        self.should_terminate = False

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

    def train(self, agent: Optional[Agent], num_episodes: int, max_episode_timesteps: int, weights_dir='weights/agents',
              agent_name='carla-agent', load_agent=False, record_dir='data/recordings', skip_frames=25):
        record_path = None
        should_record = isinstance(record_dir, str)
        should_save = isinstance(weights_dir, str)

        if agent is None:
            print('Using default agent "if available"...')
            agent = self.default_agent()

        try:
            if load_agent:
                agent.load(directory='weights_dir', filename=agent_name, environment=self)
                print('Agent loaded.')

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
        finally:
            self.close()

    def default_sensors(self) -> dict:
        """Returns a predefined dict of sensors specifications"""
        return dict(imu=SensorSpecs.imu(),
                    collision=SensorSpecs.collision_detector(),
                    camera=SensorSpecs.rgb_camera(position='top',
                                                  image_size_x=self.image_shape[1], image_size_y=self.image_shape[0],
                                                  sensor_tick=1.0 / self.fps))

    def default_agent(self) -> Agent:
        """Returns a predefined agent for this environment"""
        raise NotImplementedError('Implement this to define your own default agent!')

    def on_collision(self, event: carla.CollisionEvent, penalty=1000.0, max_impulse=400.0):
        impulse = math.sqrt(utils.vector_norm(event.normal_impulse))
        actor_type = event.other_actor.type_id
        print(f'Collision(impulse={round(impulse, 2)}, actor={actor_type})')

        if 'pedestrian' in actor_type:
            self.collision_penalty += max(penalty * impulse, penalty)
            self.should_terminate = True
        elif 'vehicle' in actor_type:
            self.collision_penalty += max(penalty / 2 * impulse, penalty)
            self.should_terminate = True
        else:
            self.collision_penalty += min(impulse, max_impulse)
            self.should_terminate = False

    def render(self, image: carla.Image, data: dict):
        env_utils.display_image(self.display, image, window_size=self.window_size)

        if self.sensors.get('sem_camera', False):
            segmentation = self.sensors['sem_camera'].convert_image(data['sem_camera'])
            env_utils.display_image(self.display, segmentation, window_size=self.window_size, blend=True)

    def debug(self, actions):
        env_utils.display_text(self.display, self.font, text=self._get_debug_text(actions), origin=(16, 12),
                               offset=(0, 16))

    def skip(self, num_frames=10):
        """Skips the given amount of frames"""
        for _ in range(num_frames):
            self.synchronous_context.tick(timeout=self.timeout)

        if num_frames > 0:
            print(f'Skipped {num_frames} frames.')

    def control_to_actions(self, control: carla.VehicleControl):
        raise NotImplementedError("Implement only if needed for pretraining.")

    def on_pre_world_step(self):
        """Called before world.tick()"""
        pass

    def on_post_world_step(self, sensors_output_data: dict):
        """Called after world.tick()."""
        pass

    @profile
    def _sync_world_step(self, actions, record_path: str = None):
        # [pre-tick updates] Apply control to update the vehicle
        self.actions_to_control(actions)
        self.vehicle.apply_control(self.control)

        self.on_pre_world_step()

        # Advance the simulation and wait for the data.
        data = self.synchronous_context.tick(timeout=self.timeout)
        image = self.sensors['camera'].convert_image(data['camera'])

        # [post-tick updates] Update world-related stuff
        self.on_post_world_step(data)
        self._update_env_state()

        # Draw the display
        if self.visualize:
            # TODO: maybe 'debug' should go before 'render or be independent
            self.render(image, data)

            if self.should_debug:
                self.debug(actions)

            pygame.display.flip()

            if isinstance(record_path, str):
                env_utils.pygame_save(self.display, record_path)

        return image

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
        # throttle and brake are mutual exclusive:
        self.control.throttle = float(actions[0]) if actions[0] > 0 else 0.0
        self.control.brake = float(-actions[0]) if actions[0] < 0 else 0.0

        self.control.steer = float(actions[1])

        # reverse could be enabled only if throttle > 0
        if self.control.throttle > 0:
            self.control.reverse = bool(actions[2] > 0)
        else:
            self.control.reverse = False

        # hand-brake active only if throttle > 0 and reverse is False
        if self.control.throttle > 0 and self.control.reverse:
            self.control.hand_brake = bool(actions[3] > 0)

    def _get_observation(self, image):
        if image is None:
            image = np.zeros(shape=self.image_shape, dtype=np.uint8)

        if image.shape != self.image_shape:
            image = env_utils.resize(image, size=self.image_size)

        observation = dict(image=(2 * image - 255.0) / 255.0,
                           vehicle_features=self._get_vehicle_features(),
                           road_features=self._get_road_features(),
                           previous_actions=self.prev_actions)

        # check for nan/inf values
        for key, value in observation.items():
            if np.isnan(value).any() or np.isinf(value).any():
                print(f'[{key}] NaN/Inf', np.sum(np.isnan(value)) + np.sum(np.isinf(value)))
                observation[key] = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
                print(not np.isnan(value).any() and (not np.isinf(value).any()))

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

    def _get_debug_text(self, actions):
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

    def _create_sensors(self):
        for name, args in self.sensors_spec.items():
            kwargs = args.copy()
            sensor_type = kwargs.pop('type')

            if sensor_type == 'sensor.other.collision':
                sensor = CollisionSensor(parent_actor=self.vehicle)
                sensor.add_callback(self.on_collision)

            elif sensor_type == 'sensor.other.imu':
                sensor = IMUSensor(parent_actor=self.vehicle)

            elif sensor_type == 'sensor.camera.rgb':
                sensor = RGBCameraSensor(parent_actor=self.vehicle, **kwargs)

            elif sensor_type == 'sensor.camera.semantic_segmentation':
                sensor = SemanticCameraSensor(parent_actor=self.vehicle, **kwargs)
            else:
                raise ValueError(f'Cannot create sensor `{sensor_type}`.')

            if name == 'world':
                raise ValueError(f'Cannot name a sensor `world` because is reserved.')

            self.sensors[name] = sensor
