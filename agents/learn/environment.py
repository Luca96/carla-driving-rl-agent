"""Contains various environments build to work with CARLA simulator."""

import cv2
import pygame

from tensorforce.environments import Environment

from agents.learn import env_utils
from agents.learn.env_utils import Profiler, profile

from worlds import World, Route, RoutePlanner, WAYPOINT_DICT
from worlds.sensors import *
from worlds.debug.graphics import HUD, CARLADebugInfo, CARLADebugInfoSmall
from worlds.tools import misc
from worlds.tools.synchronous_mode import CARLASyncContext

MAX_RADIANS = 2.0 * math.pi
MAX_SPEED = 150.0


class CarlaEnvironment(Environment):
    """A TensorForce environment designed to work with the CARLA driving simulator."""
    # actions: throttle, steer, brake, reverse (bool), hand_brake (bool)
    ACTIONS_SPEC_DICT = dict(type='float', shape=(4, ), min_value=-1.0, max_value=1.0)
    # ACTIONS_SPEC_DICT = dict(throttle=dict(type='float', shape=(1, ), min_value=0.0, max_value=1.0),
    #                          steer=dict(type='float', shape=(1, ), min_value=-1.0, max_value=1.0),
    #                          brake=dict(type='float', shape=(1, ), min_value=0.0, max_value=1.0),
    #                          reverse=dict(type='bool', shape=(1, )))

    # vehicle: speed, gear, accelerometer (x, y, z), gysroscope (x, y, z), position (x, y), destination (x, y), compass
    VEHICLE_FEATURES_SPEC_DICT = dict(type='float', shape=(14, ))

    # road: intersection (bool), junction (bool), speed_limit, traffic_light (presence + state), lane_width,
    # lane_change, left_lane, right_lane
    ROAD_FEATURES_SPEC_DICT = dict(type='float', shape=(10, ))

    # law: speed limit, traffic light (bool). traffic light state
    LAW_NORMS_SPEC_DICT = dict(type='float', shape=(3, ))

    # DEFAULT_ACTIONS = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    DEFAULT_ACTIONS = np.array([0.0, 0.0, 0.0, 0.0])
    # DEFAULT_ACTIONS = np.array([0., 0., 0.])
    # DEFAULT_ACTIONS = np.array([0, 0, 0, 0])

    # TODO: provide one (str) or many maps ([str]) to load.
    # TODO: provide map loading mode: 'random' or 'sequential'. -> a map is loaded on reset(), i.e. when an episode ends
    def __init__(self, client: carla.Client,
                 image_shape: (int, int, int),
                 map='Town03',
                 route_resolution=2.0,
                 actor_filter='vehicle.*',
                 window_size=(800, 600),
                 max_fps=60.0,
                 **kwargs):
        """
        :param client: a carla.Client instance.
        :param image_shape: ...
        :param map: the name of the map to load.
        :param route_resolution: sets the grain (resolution) of the planner path.
        :param actor_filter: use to decide which actor blueprint can be loaded.
        """
        super().__init__()
        print('env.create')
        self.client = client
        self.map = map
        self.image_shape = image_shape
        self.actor_filter = actor_filter
        self.window_size = window_size

        self.control = None
        self.steer_cache = 0.0
        self.prev_actions = None

        # graphics:
        self.world = World(self.client.get_world(),
                           window_size=self.window_size,
                           actor_filter=self.actor_filter,
                           init=False)  # prevents the call of 'start()'
        self.max_fps = max_fps
        self.clock = pygame.time.Clock()
        self.display = pygame.display.set_mode(self.window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)

        # TODO: decide if callbacks are needed, and if so decide their arguments and function
        # callbacks
        self.on_reset = None

    def states(self):
        return dict(image=dict(shape=self.image_shape),
                    vehicle_features=self.VEHICLE_FEATURES_SPEC_DICT,
                    road_features=self.ROAD_FEATURES_SPEC_DICT,
                    previous_actions=self.ACTIONS_SPEC_DICT)

    def actions(self):
        return self.ACTIONS_SPEC_DICT

    def reset(self):
        print('env.reset')
        self.control = carla.VehicleControl()
        self.prev_actions = self.DEFAULT_ACTIONS
        self.steer_cache = 0.0

        self.world.restart()
        _, image = self._world_step()

        if self.on_reset:
            self.on_reset()

        return self._get_observation(image)

    def on_reset_event(self, callback):
        self.on_reset = callback

    def execute(self, actions):
        self.prev_actions = actions
        self._actions_to_control(actions)
        self.world.apply_control(self.control)

        reward, image = self._world_step()
        terminal = (self.world.num_collisions > 0) or (self.world.distance_to_destination() < 10)
        next_state = self._get_observation(image)

        return next_state, terminal, reward

    def close(self):
        print('env.close')
        super().close()
        self.world.destroy()

    def _world_step(self):
        self.clock.tick_busy_loop(self.max_fps)

        reward = self.world.tick(self.clock)
        image = self.world.render(self.display)
        pygame.display.flip()

        return reward, image

    def _get_observation(self, image):
        # assert image is not None
        if image is None:
            print('_get_observation --> None')
            image = np.zeros(shape=self.image_shape, dtype=np.uint8)

        shape = (self.image_shape[1], self.image_shape[0])

        return dict(image=cv2.resize(image, dsize=shape, interpolation=cv2.INTER_CUBIC),
                    vehicle_features=self.world.get_vehicle_features(),
                    road_features=self.world.get_road_features(),
                    previous_actions=self.prev_actions)

    # def _actions_to_control(self, actions):
    #     print(actions)
    #
    #     # Throttle
    #     if actions[0] == 2:
    #         self.control.throttle = 1.0
    #
    #         # Reverse
    #         if actions[3] == 2:
    #             self.control.reverse = 1.0
    #         else:
    #             self.control.reverse = 0.0
    #
    #     else:
    #         self.control.throttle = 0.0
    #
    #     # Steer
    #     if actions[1] == 2:
    #         self.control.steer = 1.0
    #     elif actions[1] == 1:
    #         self.control.steer = 0.0
    #     else:
    #         self.control.steer = -1.0
    #
    #     # Brake
    #     if actions[2] == 2:
    #         self.control.brake = 1.0
    #     else:
    #         self.control.brake = 0.0

    def _actions_to_control(self, actions):
        # 1
        # self.control.throttle = float((actions[0] + 1) / 2.0)
        # self.control.steer = float(actions[1])
        # self.control.brake = float((actions[2] + 1) / 2.0)
        # self.control.reverse = bool(actions[3] > 0)  # False
        # self.control.hand_brake = bool(actions[4] > 0)
        # print(self.control)
        # return

        # 2
        steer_increment = 5e-4 * self.clock.get_time()

        # Throttle
        if actions[0] < 0:
            self.control.throttle = 0.0
            self.control.brake = 1.0
        else:
            self.control.throttle = 1.0
            self.control.brake = 0.0

        # Steer
        if actions[1] < -0.33:
            # turn left
            # self.control.throttle = 1.0
            # self.control.steer = -1 * steer_increment
            # self.control.steer = float(actions[0])

            # self.steer_cache = max(-1, self.steer_cache - steer_increment)
            # self.control.steer = self.steer_cache
            self.control.steer = -0.5
        elif actions[1] > 0.33:
            # turn right
            # self.control.throttle = 1.0
            # self.control.steer = 1 * steer_increment
            # self.control.steer = float(actions[0])

            # self.steer_cache = min(1, self.steer_cache + steer_increment)
            # self.control.steer = self.steer_cache
            self.control.steer = 0.5
        else:
            # go straight
            # self.control.throttle = 1.0
            self.control.steer = 0

        # self.control.brake = float((actions[1] + 1) / 2.0)
        # self.control.brake = 0.0 if actions[2] < 0 else 1.0
        self.control.reverse = bool(actions[2] < 0)


# TODO: the best way to customize the environment's states, actions, reward and so on it to make a base-abstract
#  class and then have the specific behaviour as a subclass
class SynchronousCARLAEnvironment(Environment):
    """A TensorForce Environment for the CARLA driving simulator.
        - This environment is "synchronized" with the server, meaning that the server waits for a client tick.
    """
    # States and actions specifications:
    # Actions: skill, throttle, steer, brake, reverse (bool), hand_brake (bool)
    ACTIONS_SPEC = dict(type='float', shape=(6,), min_value=-1.0, max_value=1.0)
    DEFAULT_ACTIONS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # A "skill" is a high-level action
    SKILLS = {0: 'idle',     1: 'brake',
              2: 'forward',  3: 'forward left',  4: 'forward right',
              5: 'backward', 6: 'backward left', 7: 'backward right'}

    # Vehicle: speed, gear, accelerometer (x, y, z), gyroscope (x, y, z), position (x, y), destination (x, y), compass
    VEHICLE_FEATURES_SPEC = dict(type='float', shape=(14,))

    # Road: intersection (bool), junction (bool), speed_limit, traffic_light (presence + state), lane_width,
    #       lane_change, left_lane, right_lane
    ROAD_FEATURES_SPEC = dict(type='float', shape=(10,))

    # Law: speed limit, traffic light (bool). traffic light state
    LAW_NORMS_SPEC = dict(type='float', shape=(3,))

    # default sensors specification
    # TODO: sensor class specification to shorten the definition of sensors? At least for cameras=
    # DEFAULT_SENSORS = dict(imu=True,
    #                        # collision_detector=True,
    #                        rgb_camera=dict(transform=carla.Transform(carla.Location(x=-5.5, z=2.5),
    #                                                                  carla.Rotation(pitch=8.0)),
    #                                        attachment_type=carla.AttachmentType.SpringArm,
    #                                        color_converter=carla.ColorConverter.Raw,
    #                                        attributes=dict(image_size_x=200, image_size_y=150)))
    DEFAULT_SENSORS = dict(imu=dict(type='sensor.other.imu'),
                           collision=dict(type='sensor.other.collision'),
                           camera=dict(type='sensor.camera.rgb',
                                       transform=carla.Transform(carla.Location(x=-5.5, z=2.5),
                                                                 carla.Rotation(pitch=8.0)),
                                       attachment_type=carla.AttachmentType.SpringArm,
                                       color_converter=carla.ColorConverter.Raw,
                                       attributes=dict(image_size_x=200, image_size_y=150)))

    # TODO: add a loading map functionality (specified or at random) - load_map
    # TODO: add debug flag(s)
    # TODO: add support for custom state specification as well as actions specification.
    # TODO: add type assertions
    # TODO: add recording support?
    # TODO: add possibility to specify the spawn point ('fixed' or 'random')
    def __init__(self, address='localhost', port=2000, timeout=2.0, image_shape=(200, 150, 3), window_size=(800, 600),
                 vehicle_filter='vehicle.*', sensors: dict = None, route_resolution=2.0, fps=30.0, visualize=True,
                 debug=False,):
        super().__init__()
        self.client = env_utils.get_client(address, port, timeout)
        self.world: carla.World = self.client.get_world()   # TODO: make a wrapper -> CARLAWorld
        self.map: carla.Map = self.world.get_map()
        self.debug_helper: carla.DebugHelper = self.world.debug
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

        # vehicle sensors suite
        self.sensors_spec = sensors if sensors is not None else self.DEFAULT_SENSORS
        self.sensors = dict()

        # high-level path planning
        self.route = Route(planner=RoutePlanner(map=self.map, sampling_resolution=route_resolution))
        self.spawn_point = None
        self.destination = None

        # weather
        # TODO: add weather support

        # visualization + debug stuff
        self.image_shape = image_shape
        self.image_size = (image_shape[1], image_shape[0])
        self.fps = fps
        self.visualize = visualize
        self.debug = debug
        self.clock = pygame.time.Clock()

        if self.visualize:
            self.window_size = window_size
            self.font = env_utils.get_font()
            self.display = env_utils.get_display(window_size)

        if self.debug:
            self.debug_info = CARLADebugInfoSmall(width=window_size[0], height=window_size[1], environment=self)

        # variables for reward computation
        self.last_location = None
        self.travelled_distance = 0.0
        self.num_collisions = 0

        # event callbacks
        # TODO: add more callbacks/events if necessary
        self.on_reset = None

    def states(self):
        # TODO: also consider previous vehicle control?
        return dict(image=dict(shape=self.image_shape),
                    vehicle_features=self.VEHICLE_FEATURES_SPEC,
                    road_features=self.ROAD_FEATURES_SPEC,
                    previous_actions=self.ACTIONS_SPEC)

    def actions(self):
        return self.ACTIONS_SPEC

    def reset(self):
        print('env.reset')
        self._reset_world()

        # reset actions
        self.control = carla.VehicleControl()
        self.prev_actions = self.DEFAULT_ACTIONS

        return self._get_observation(image=None)

    # TODO: provide support for custom reward functions
    def reward(self, time_cost=-1.0, b=-1000.0, c=2.0, d=2.0):
        """Agent's reward function"""
        # TODO: include a penalty for law compliance: exceeding the speed limit, red traffic light...

        # Collision term (n. of collision between two tick): do as few collision as possible
        collision_penalty = b * self.num_collisions

        # Direction term: alignment of the vehicle's heading direction with the waypoint's forward vector
        closest_waypoint = self.route.closest_path.waypoint
        similarity = utils.cosine_similarity(self.vehicle.get_transform().get_forward_vector(),  # heading direction
                                             closest_waypoint.transform.get_forward_vector())
        speed = utils.speed(self.vehicle)

        if similarity > 0:
            direction_penalty = (speed + 1) * similarity  # speed + 1, to avoid 0 speed
        else:
            direction_penalty = (speed + 1) * similarity * d

        if speed < 10:
            direction_penalty -= 1

        if self.travelled_distance <= self.route.size:
            efficiency_term = 0
        else:
            efficiency_term = -(self.travelled_distance - self.route.size) - self.route.distance_to_destination()

            # travelled more than route size, zero direction_penalty if positive
            if direction_penalty > 0:
                direction_penalty = 0

        # Speed-limit compliance:
        speed_limit = self.vehicle.get_speed_limit()

        if speed <= speed_limit:
            speed_penalty = 0
        else:
            speed_penalty = c * (speed_limit - speed)

        return time_cost + collision_penalty + efficiency_term + direction_penalty + speed_penalty

    @profile
    def execute(self, actions):
        self.prev_actions = actions

        # https://stackoverflow.com/questions/20165492/pygame-window-not-responding-after-a-few-seconds
        pygame.event.get()
        # pygame.event.pump()
        self.clock.tick()

        image = self._sync_world_step(self.synchronous_context, actions)

        # TODO: handle num_collision properly, i.e. differentiate between collision type
        reward = self.reward()
        terminal = (self.num_collisions > 0) or (self.route.distance_to_destination() <= 10)
        next_state = self._get_observation(image)

        # Action penalty: penalise actions according to current skill (i.e. actions[0])
        action_penalty = self._action_penalty(actions)
        if action_penalty > 0:
            reward = -abs(reward) - action_penalty

        # Reset collision count
        self.num_collisions = 0

        return next_state, terminal, reward

    def close(self):
        print('env.close')
        super().close()

        if self.vehicle:
            self.vehicle.destroy()

        for sensor in self.sensors.values():
            sensor.destroy()

    def _reset_world(self):
        print('env.reset_world')
        # init actor
        spawn_point = env_utils.random_spawn_point(self.map)

        if self.vehicle is None:
            blueprint = env_utils.random_blueprint(self.world, actor_filter=self.vehicle_filter)
            self.vehicle: carla.Vehicle = env_utils.spawn_actor(self.world, blueprint, spawn_point)

            self._create_sensors()
            self.synchronous_context = CARLASyncContext(self.world, self.sensors, fps=self.fps)
        else:
            # TODO: cannot change blueprint without re-spawning the actor!
            self.vehicle.set_transform(spawn_point)

        self.spawn_point = spawn_point
        self.last_location: carla.Location = spawn_point.location
        self.destination: carla.Location = env_utils.random_spawn_point(self.map,
                                                                        different_from=spawn_point.location).location
        # plan path
        self.route.plan(origin=self.spawn_point.location, destination=self.destination)

        # reset reward variables
        self.travelled_distance = 0.0
        self.num_collisions = 0

        # TODO: reset sensors?

    @profile
    def _sync_world_step(self, sync_mode: CARLASyncContext, actions):
        # [pre-tick updates] Apply control to update the vehicle
        self._actions_to_control(actions)
        self.vehicle.apply_control(self.control)

        # self.route.draw_route(self.debug_helper, life_time=1.0)
        # self.route.draw_closest_waypoint(self.debug_helper, self.vehicle.get_location(), life_time=1.0)

        # Advance the simulation and wait for the data.
        data = sync_mode.tick(timeout=2.0)
        image = self.sensors['camera'].convert_image(data['camera'])

        # [post-tick updates] Update world-related stuff
        self.route.update_closest_waypoint(location=self.vehicle.get_location())
        self._update_travelled_distance()

        # Draw the display
        if self.visualize:
            if (image.shape[1], image.shape[0]) != self.window_size:
                image = env_utils.resize(image, size=self.window_size)

            env_utils.display_image(self.display, image)

            # TODO: debug stuff are too slow, more than 0.1ms!!!
            # if self.debug:
            #     self.debug_info.on_world_tick(snapshot=data['world'])
            #     self.debug_info.tick(self.clock)
            #     self.debug_info.render(self.display)

            env_utils.display_text(self.display, self.font,
                                   text=['%d FPS' % self.clock.get_fps(),
                                         '',
                                         'Throttle: %.2f' % self.control.throttle,
                                         'Steer: %.2f' % self.control.steer,
                                         'Brake: %.2f' % self.control.brake,
                                         'Reverse: %s' % ('T' if self.control.reverse else 'F'),
                                         'Hand brake: %s' % ('T' if self.control.hand_brake else 'F'),
                                         'Gear: %s' % {-1: 'R', 0: 'N'}.get(self.control.gear)],
                                   origin=(16, 12), offset=(0, 16))
            pygame.display.flip()

        return image

    def _update_travelled_distance(self):
        location1 = self.last_location
        location2 = self.vehicle.get_location()

        self.travelled_distance += misc.compute_distance(location1, location2)
        self.last_location = location2

    def _actions_to_control(self, actions):
        self.control.steer = float(actions[2])

        # throttle and brake are mutual exclusive:
        self.control.throttle = float(actions[1]) if actions[1] > 0 else 0.0
        self.control.brake = float(-actions[1]) if actions[1] < 0 else 0.0

        # reverse could be enabled only if throttle > 0
        if self.control.throttle > 0:
            self.control.reverse = bool(actions[4] > 0)
        else:
            self.control.reverse = False

        # hand-brake active only if throttle > 0 and reverse is False
        if self.control.throttle > 0 and self.control.reverse:
            self.control.hand_brake = bool(actions[5] > 0)

    def get_skill_name(self):
        skill = env_utils.scale(self.prev_actions[0])
        return self.SKILLS[int(skill)]

    def _action_penalty(self, actions, alpha=1, epsilon=0.01):
        # epsilon accounts for little noise in the actions
        skill = int(env_utils.scale(actions[0]))
        penalty = 0.0

        throttle = self.control.throttle
        steer = self.control.steer
        brake = self.control.brake
        reverse = self.control.reverse
        hand_brake = self.control.hand_brake

        if skill == 1:
            # brake => T = 0, B > 0, H = True or H = False
            penalty += 0 if throttle <= epsilon else alpha
            penalty += 0 if brake >= epsilon else alpha
        elif skill == 2:
            # forward (don't turn) => T > 0, S = 0, B = 0, R = False, H = False
            penalty += 0 if throttle >= epsilon else alpha
            penalty += 0 if -epsilon <= steer <= +epsilon else alpha
            penalty += 0 if brake <= epsilon else alpha
            penalty += 0 if reverse is False else alpha
            penalty += 0 if hand_brake is False else alpha
        elif skill == 3:
            # forward + turn left => T > 0, S < 0, B = 0, R = False, H = False
            penalty += 0 if throttle >= epsilon else alpha
            penalty += 0 if steer <= -epsilon else alpha
            penalty += 0 if brake <= epsilon else alpha
            penalty += 0 if reverse is False else alpha
            penalty += 0 if hand_brake is False else alpha
        elif skill == 4:
            # forward + turn right => T > 0, S > 0, B = 0, R = False, H = False
            penalty += 0 if throttle >= epsilon else alpha
            penalty += 0 if steer >= epsilon else alpha
            penalty += 0 if brake <= epsilon else alpha
            penalty += 0 if reverse is False else alpha
            penalty += 0 if hand_brake is False else alpha
        elif skill == 5:
            # backward (don't turn) => T > 0, S = 0, B = 0, R = True, H = False
            penalty += 0 if throttle >= epsilon else alpha
            penalty += 0 if -epsilon <= steer <= +epsilon else alpha
            penalty += 0 if brake <= epsilon else alpha
            penalty += 0 if reverse is True else alpha
            penalty += 0 if hand_brake is False else alpha
        elif skill == 6:
            # backward + turn left => T > 0, S < 0, B = 0, R = True, H = False
            penalty += 0 if throttle >= epsilon else alpha
            penalty += 0 if steer <= -epsilon else alpha
            penalty += 0 if brake <= epsilon else alpha
            penalty += 0 if reverse is True else alpha
            penalty += 0 if hand_brake is False else alpha
        elif skill == 7:
            # backward + turn right => T > 0, S > 0, B = 0, R = True, H = False
            penalty += 0 if throttle >= epsilon else alpha
            penalty += 0 if steer >= epsilon else alpha
            penalty += 0 if brake <= epsilon else alpha
            penalty += 0 if reverse is True else alpha
            penalty += 0 if hand_brake is False else alpha
        else:
            # idle/stop (skill == 0) => T = 0, S = 0
            penalty += 0 if throttle <= epsilon else alpha
            penalty += 0 if -epsilon <= steer <= +epsilon else alpha

        return penalty

    def _get_observation(self, image):
        if image is None:
            image = np.zeros(shape=self.image_shape, dtype=np.uint8)

        if image.shape != self.image_shape:
            image = env_utils.resize(image, size=self.image_size)

        return dict(image=image,
                    vehicle_features=self._get_vehicle_features(),
                    road_features=self._get_road_features(),
                    previous_actions=self.prev_actions)

    def _get_vehicle_features(self):
        """Returns a dict(speed, position, destination, compass) representing the vehicle location state"""
        t = self.vehicle.get_transform()

        imu_sensor = self.sensors['imu']
        gyroscope = imu_sensor.gyroscope
        accelerometer = imu_sensor.accelerometer

        return [  # Speed and gear
            utils.speed(self.vehicle),
            self.vehicle.get_control().gear,
            self.vehicle.get_control().reverse,
            # Accelerometer:
            accelerometer[0],
            accelerometer[1],
            accelerometer[2],
            # Gyroscope:
            gyroscope[0],
            gyroscope[1],
            gyroscope[2],
            # Location  # TODO: change with 'gnss' measurement??
            t.location.x,
            t.location.y,
            # Destination:
            self.destination.x,
            self.destination.y,
            # TODO: add 'light_state' -> release 0.9.8
            # Compass:
            math.radians(imu_sensor.compass)]

    def _get_road_features(self):
        waypoint = self.map.get_waypoint(self.vehicle.get_location())

        # TODO: move the above 3 features to vehicle-features?
        speed_limit = self.vehicle.get_speed_limit()
        is_at_traffic_light = self.vehicle.is_at_traffic_light()

        if is_at_traffic_light:
            traffic_light_state = self.vehicle.get_traffic_light_state()
        else:
            traffic_light_state = carla.TrafficLightState.Unknown

        return [waypoint.is_intersection,
                waypoint.is_junction,
                waypoint.lane_width,
                speed_limit,
                # Traffic light:
                is_at_traffic_light,
                WAYPOINT_DICT['traffic_light'][traffic_light_state],
                # Lane:
                WAYPOINT_DICT['lane_type'][waypoint.lane_type],
                WAYPOINT_DICT['lane_change'][waypoint.lane_change],
                WAYPOINT_DICT['lane_marking_type'][waypoint.left_lane_marking.type],
                WAYPOINT_DICT['lane_marking_type'][waypoint.right_lane_marking.type]]

    def _create_sensors(self):
        for name, kwargs in self.sensors_spec.items():
            sensor_type = kwargs.pop('type')

            if sensor_type == 'sensor.other.collision':
                sensor = CollisionSensor(parent_actor=self.vehicle,
                                         hud=self.debug_info,
                                         callback=lambda _: print('collision'))

            elif sensor_type == 'sensor.other.imu':
                sensor = IMUSensor(parent_actor=self.vehicle)

            elif sensor_type == 'sensor.camera.rgb':
                sensor = RGBCameraSensor(parent_actor=self.vehicle, **kwargs)
            else:
                raise ValueError(f'Cannot create sensor `{sensor_type}`.')

            if name == 'world':
                raise ValueError(f'Cannot name a sensor `world` because is reserved.')

            self.sensors[name] = sensor
