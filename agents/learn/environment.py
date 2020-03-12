"""Contains various environments build to work with CARLA simulator."""

import cv2
import math
import carla
import numpy as np
import pygame

from tensorforce.environments import Environment
from worlds import World
from worlds.debug import HUD

MAX_RADIANS = 2.0 * math.pi
MAX_SPEED = 150.0


class CarlaEnvironment(Environment):
    """TensorForce Environment wrapper suitable for working with CARLA simulator."""
    ACTIONS_SPEC_DICT = dict(type='float', shape=(5, ), min_value=-1.0, max_value=1.0)
    # ACTIONS_SPEC_DICT = dict(throttle=dict(type='float', shape=(1, ), min_value=0.0, max_value=1.0),
    #                          steer=dict(type='float', shape=(1, ), min_value=-1.0, max_value=1.0),
    #                          brake=dict(type='float', shape=(1, ), min_value=0.0, max_value=1.0),
    #                          reverse=dict(type='bool', shape=(1, )),
    #                          hand_brake=dict(type='bool', shape=(1, )))

    VEHICLE_FEATURES_SPEC_DICT = dict(type='float', shape=(6, ))
    # VEHICLE_FEATURES_SPEC_DICT = dict(speed=dict(type='float', shape=(1, ), min_value=0.0, max_value=MAX_SPEED),
    #                                   position=dict(type='float', shape=(2, )),
    #                                   destination=dict(type='float', shape=(2, )),
    #                                   compass=dict(type='float', shape=(1, ), min_value=0.0, max_value=MAX_RADIANS))

    ROAD_FEATURES_SPEC_DICT = dict(type='float', shape=(7, ))
    # ROAD_FEATURES_SPEC_DICT = dict(is_intersection=dict(type='bool', shape=(1, )),
    #                                is_junction=dict(type='bool', shape=(1, )),
    #                                lane_width=dict(type='float', shape=(1, ), min_value=0.0, max_value=5.0),
    #                                lane_type=dict(type='int', shape=(1, ), num_states=22),
    #                                lane_change=dict(type='int', shape=(1, ), num_states=4),
    #                                left_lane_type=dict(type='int', shape=(1, ), num_states=11),
    #                                right_lane_type=dict(type='int', shape=(1, ), num_states=11))
    # TODO: add 'speed_limit', 'traffic_light_presence', 'traffic_light_state'

    # speed limit, traffic light. traffic light state
    LAW_NORMS_SPEC_DICT = dict(type='float', shape=(3, ))

    # DEFAULT_ACTIONS = dict(throttle=0.0, steer=0.0, brake=0.0, reverse=False, hand_brake=False)
    DEFAULT_ACTIONS = np.array([0., 0., 0., 0., 0.])

    def __init__(self, world: World, display: pygame.display, image_shape: tuple, max_fps=120):
        """Arguments:
            @:arg worlds: a utils.World instance,
            @:arg image_shape: the shape (w, h, c) of the image observations from camera sensor,
            @:arg update_callback: the callback that updates the worlds object during the 'execute' method by applying
                                   the computed actions.
            @:arg: display: a pygame.display instance, used for rendering purpose.
        """
        # TODO: consider to add a 'seed' argument to make randomness deterministic!
        super().__init__()
        self.world = world
        self.control = carla.VehicleControl()
        self.image_shape = image_shape
        self.prev_actions = self.DEFAULT_ACTIONS

        # Variable for updating the worlds
        self.clock = pygame.time.Clock()
        self.max_fps = max_fps
        self.display = display

    def states(self):
        return dict(image=dict(shape=self.image_shape),
                    vehicle_features=self.VEHICLE_FEATURES_SPEC_DICT,
                    road_features=self.ROAD_FEATURES_SPEC_DICT,
                    previous_actions=self.ACTIONS_SPEC_DICT)

    def actions(self):
        return self.ACTIONS_SPEC_DICT

    def reset(self):
        print('env.reset')
        self.prev_actions = self.DEFAULT_ACTIONS

        # TODO: fix worlds restart!!!
        self.world.restart()
        # self.world.num_collisions = 0

        _, image = self._world_step()
        return self._get_observation(image)

    def execute(self, actions):
        self.prev_actions = actions
        self._actions_to_control(actions)
        self.world.player.apply_control(self.control)  # TODO: refactor -> hide .player

        reward, image = self._world_step()
        terminal = (self.world.num_collisions > 0) or (self.world.distance_to_destination() < 2)  # hyperparameter
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
        # self.control.throttle = float(actions['throttle'])
        # self.control.steer = float(actions['steer'])
        # self.control.brake = float(actions['brake'])
        # self.control.reverse = bool(actions['reverse'])
        # self.control.hand_brake = bool(actions['hand_brake'])

        # 2
        # self.control.throttle = float((actions[0] + 1) / 2.0)
        # self.control.steer = float(actions[1])
        # self.control.brake = float((actions[2] + 1) / 2.0)
        # self.control.reverse = bool(actions[3] < 0)
        # self.control.hand_brake = bool(actions[4] < 0)

        # 3
        if actions[0] > 0:
            # forward
            self.control.throttle = float(actions[0])
            # self.control.reverse = False
        else:
            # backward
            self.control.throttle = float(-actions[0])
            # self.control.reverse = True

        # self.control.throttle = 0.2
        self.control.steer = float(actions[1])
        self.control.brake = float((actions[2] + 1) / 2.0)
        # self.control.reverse = bool(actions[3] < 0)  # False
        # self.control.hand_brake = bool(actions[4] < 0)
        # print(self.control)


class CarlaEnv(Environment):
    """A TensorForce environment designed to work with the CARLA driving simulator."""
    # actions: throttle, steer, brake, reverse (bool), hand_brake (bool)
    ACTIONS_SPEC_DICT = dict(type='float', shape=(5,), min_value=-1.0, max_value=1.0)

    # vehicle: speed, position (x, y), destination (x, y), compass
    VEHICLE_FEATURES_SPEC_DICT = dict(type='float', shape=(6,))

    # TODO: add 'speed_limit', 'traffic_light_presence', 'traffic_light_state'
    # road: intersection (bool), junction (bool), lane_width, lane_change, left_lane, right_lane

    ROAD_FEATURES_SPEC_DICT = dict(type='float', shape=(7,))
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
        self.control.throttle = float((actions[0] + 1) / 2.0)
        self.control.steer = float(actions[1])
        self.control.brake = float((actions[2] + 1) / 2.0)
        # self.control.reverse = bool(actions[3] < 0)  # False
        # self.control.hand_brake = bool(actions[4] < 0)
        # print(self.control)
