"""Contains various environments build to work with CARLA simulator."""

import cv2
import math
import carla
import numpy as np
import pygame
import tensorflow as tf

from tensorforce.environments import Environment

from worlds import World
from worlds.debug import HUD
from agents.learn import env_utils

MAX_RADIANS = 2.0 * math.pi
MAX_SPEED = 150.0


class CarlaEnvironment(Environment):
    """A TensorForce environment designed to work with the CARLA driving simulator."""
    # actions: throttle, steer, brake, reverse (bool), hand_brake (bool)
    ACTIONS_SPEC_DICT = dict(type='float', shape=(5, ), min_value=-1.0, max_value=1.0)

    # vehicle: speed, accelerometer (x, y, z), gysroscope (x, y, z), position (x, y), destination (x, y), compass
    VEHICLE_FEATURES_SPEC_DICT = dict(type='float', shape=(12, ))

    # road: intersection (bool), junction (bool), speed_limit, traffic_light (presence + state), lane_width,
    # lane_change, left_lane, right_lane
    ROAD_FEATURES_SPEC_DICT = dict(type='float', shape=(10, ))

    # law: speed limit, traffic light (bool). traffic light state
    LAW_NORMS_SPEC_DICT = dict(type='float', shape=(3, ))

    DEFAULT_ACTIONS = np.array([0., 0., 0., 0., 0.])

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
                           hud=HUD(width=self.window_size[0], height=self.window_size[1]),
                           actor_filter=self.actor_filter,
                           init=False)  # prevents the call of 'start()'
        self.max_fps = max_fps
        self.clock = pygame.time.Clock()
        self.display = pygame.display.set_mode(self.window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)

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

        return self._get_observation(image)

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

    def _actions_to_control(self, actions):
        # 1
        # self.control.throttle = float((actions[0] + 1) / 2.0)
        # self.control.steer = float(actions[1])
        # self.control.brake = float((actions[2] + 1) / 2.0)
        # self.control.reverse = bool(actions[3] < 0)  # False
        # self.control.hand_brake = bool(actions[4] < 0)
        # print(self.control)

        # 2
        steer_increment = 5e-4 * self.clock.get_time()

        # Throttle
        if actions[0] < -0.33:
            self.control.throttle = 0.3
        elif actions[0] > 0.33:
            self.control.throttle = 0.9
        else:
            self.control.throttle = 0.5

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
        self.control.brake = 0.0 if actions[2] < 0 else float(actions[2])
        # self.control.reverse = bool(actions[2] < 0)


class CarlaCompressImageEnv(CarlaEnvironment):
    """Uses a convolutional NN to compress image observations."""

    def __init__(self, backend='nasnet_mobile', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.network = env_utils.get_image_compressor_model(backend)
        self.image_shape = (224, 224, 3)  # default for pre-trained keras.applications's models
        self.output_shape = self.network.output.shape[1]

    def states(self):
        return dict(image_features=dict(type='float', shape=(self.output_shape, )),
                    vehicle_features=self.VEHICLE_FEATURES_SPEC_DICT,
                    road_features=self.ROAD_FEATURES_SPEC_DICT,
                    previous_actions=self.ACTIONS_SPEC_DICT)

    def _get_observation(self, image):
        if image is None:
            print('_get_observation --> None')
            image = np.zeros(shape=self.image_shape, dtype=np.uint8)
        else:
            w, h = self.image_shape[:2]
            image = cv2.resize(image, dsize=(h, w), interpolation=cv2.INTER_CUBIC)

        image = np.expand_dims(image / 255.0, axis=0)  # add batch dimension
        features = self.network.predict(image)

        return dict(image_features=features,
                    vehicle_features=self.world.get_vehicle_features(),
                    road_features=self.world.get_road_features(),
                    previous_actions=self.prev_actions)
