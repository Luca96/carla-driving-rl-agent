import os
import carla
import pygame
import random
import numpy as np

from gym import spaces

from rl import utils
from rl import ThreeCameraCARLAEnvironment, CARLAEvent
from rl.environments.carla.tools import utils as carla_utils
from rl.environments.carla import env_utils

from typing import Dict, Tuple, Optional, Union


class CARLAEnv(ThreeCameraCARLAEnvironment):
    ACTION = dict(space=spaces.Box(low=-1.0, high=1.0, shape=(2,)), default=np.zeros(shape=2, dtype=np.float32))

    VEHICLE_FEATURES = dict(space=spaces.Box(low=0.0, high=1.0, shape=(4,)),
                            default=np.zeros(shape=4, dtype=np.float32))

    NAVIGATION_FEATURES = dict()
    ROAD_FEATURES = dict(space=spaces.Box(low=0.0, high=1.0, shape=(9,)), default=np.zeros(shape=9, dtype=np.float32))

    def __init__(self, *args, stack_depth=False, collision_penalty=1000.0, info_every=1, time_horizon=4,
                 past_obs_freq=4, throttle_as_desired_speed=True, num_waypoints_for_feature=5,
                 range_controls: Optional[Dict[str, Tuple[float, float]]] = None, random_weathers: list = None,
                 random_towns: list = None, record_path: str = None, **kwargs):
        """
        :param stack_depth: if true the depth-image from the depth camera sensor will be stacked along the channel
                            dimension of the image, resulting in an image with an additional channel (e.g. 3 + 1 = 4)
        :param collision_penalty: how much the agent should be penalized for colliding with other objects.
        :param info_every: how frequently in terms of steps, the additional information should be gathered.
        :param range_controls: optional dict used to specify the range for each vehicle's control.
        :param time_horizon: how much observations to consider as a single one (suitable for RNN processing)
        :param past_obs_freq: how often (in terms of steps) to consider an observation as a past observation.
        :param num_waypoints_for_feature: how many waypoints to consider for the `navigation` feature vector.
        :param random_weathers: list of carla.WeatherParameters which are sampled at each environment reset.
        :param random_towns: list of town's names, which town is loaded at each environment reset.
        """
        assert info_every >= 1
        assert time_horizon >= 1
        assert past_obs_freq >= 1
        assert num_waypoints_for_feature >= 1

        image_shape = kwargs.pop('image_shape', (90, 120, 3))

        if stack_depth:
            self.stack_depth = True
            image_shape = (image_shape[0], image_shape[1], image_shape[2] + 1)
        else:
            self.stack_depth = False

        super().__init__(*args, image_shape=image_shape, **kwargs)

        self.penalty = collision_penalty
        self.next_waypoint = None
        self.info_every = info_every
        self.interpret_throttle_as_desired_speed = throttle_as_desired_speed

        # definition of `navigation` feature:
        self.num_waypoints = num_waypoints_for_feature
        self.NAVIGATION_FEATURES['space'] = spaces.Box(low=0.0, high=25.0, shape=(self.num_waypoints,))
        self.NAVIGATION_FEATURES['default'] = np.zeros(shape=self.num_waypoints, dtype=np.float32)

        # statistics
        self.episode = -1
        self.timestep = 0
        self.total_reward = 0.0

        self.range_controls = {} if range_controls is None else range_controls
        self.info_buffer = {k: [] for k in self.info_space.spaces.keys()}

        # time horizon and past obs:
        self.time_horizon = time_horizon
        self.past_obs_freq = past_obs_freq

        # init the past observations list with t empty (default) observations
        # NOTE: the last obs is always the current (most recent) one
        self.past_obs = self._init_past_obs()

        # Random weather:
        if isinstance(random_weathers, list):
            self.should_sample_weather = True
            self.weather_set = random_weathers

            for w in random_weathers:
                assert isinstance(w, carla.WeatherParameters)
        else:
            self.should_sample_weather = False

        # Random town:
        if random_towns is None:
            self.should_sample_town = False

        elif isinstance(random_towns, list):
            if len(random_towns) == 0:
                self.should_sample_town = False
            else:
                self.should_sample_town = True
                self.town_set = random_towns

        # Record (same images)
        if record_path is None:
            self.should_record = False
        else:
            self.should_record = True
            self.record_path = utils.makedir(record_path)

    def define_sensors(self) -> dict:
        from rl import SensorSpecs
        return dict(collision=SensorSpecs.collision_detector(callback=self.on_collision),
                    imu=SensorSpecs.imu(),
                    front_camera=SensorSpecs.rgb_camera(position='on-top2', attachment_type='Rigid',
                                                                 image_size_x=self.image_size[0],
                                                                 image_size_y=self.image_size[1],
                                                                 sensor_tick=self.tick_time),
                    left_camera=SensorSpecs.rgb_camera(position='lateral-left', attachment_type='Rigid',
                                                                image_size_x=self.image_size[0],
                                                                image_size_y=self.image_size[1],
                                                                sensor_tick=self.tick_time),
                    right_camera=SensorSpecs.rgb_camera(position='lateral-right', attachment_type='Rigid',
                                                                 image_size_x=self.image_size[0],
                                                                 image_size_y=self.image_size[1],
                                                                 sensor_tick=self.tick_time))

    @property
    def observation_space(self) -> spaces.Space:
        return spaces.Dict(road=self.ROAD_FEATURES['space'], vehicle=self.VEHICLE_FEATURES['space'],
                           image=self.image_space, navigation=self.NAVIGATION_FEATURES['space'])

    @property
    def info_space(self) -> spaces.Space:
        space: spaces.Dict = super().info_space

        return spaces.Dict(episode=spaces.Discrete(n=1), timestep=spaces.Discrete(n=1),
                           total_reward=spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
                           reward=spaces.Box(low=-np.inf, high=np.inf, shape=(1,)), **space.spaces)

    def actions_to_control(self, actions):
        """Converts the given actions to vehicle's control"""
        self.control.throttle = float(actions[0]) if actions[0] > 0 else 0.0
        self.control.brake = float(-actions[0]) if actions[0] < 0 else 0.0
        self.control.steer = float(actions[1])
        self.control.hand_brake = False
        self.control.reverse = False

        if self.interpret_throttle_as_desired_speed:
            desired_speed = (float(actions[0]) + 1.0) / 2
            desired_speed *= 100.0
            current_speed = carla_utils.speed(self.vehicle)

            if current_speed == desired_speed:
                self.control.throttle = 0.0
                self.control.brake = 0.0

            elif current_speed > desired_speed:
                # brake
                self.control.throttle = 0.0
                self.control.brake = (current_speed - desired_speed) / 100.0
            else:
                # accelerate
                self.control.brake = 0.0
                self.control.throttle = (desired_speed - current_speed) / 100.0
        else:
            if carla_utils.speed(self.vehicle) < 10.0:
                self.control.brake = 0.0

        if 'throttle' in self.range_controls:
            throttle = self.range_controls['throttle']
            self.control.throttle = utils.clip(self.control.throttle, min_value=throttle[0], max_value=throttle[1])

        if 'brake' in self.range_controls:
            brake = self.range_controls['brake']
            self.control.brake = utils.clip(self.control.brake, min_value=brake[0], max_value=brake[1])

        if 'steer' in self.range_controls:
            steer = self.range_controls['steer']
            self.control.steer = utils.clip(self.control.steer, min_value=steer[0], max_value=steer[1])

    def reward(self, *args, respect_speed_limit=False, **kwargs) -> float:
        """Reward function"""
        speed = carla_utils.speed(self.vehicle)
        dw = self.route.distance_to_next_waypoint()

        if self.collision_penalty > 0.0:
            self.should_terminate = True
            return -self.collision_penalty

        if respect_speed_limit:
            speed_limit = self.vehicle.get_speed_limit()

            if speed > speed_limit:
                return speed_limit - speed

        r = speed * self.similarity

        if r != 0.0:
            r /= max(1.0, (dw / 2.0)**2)

        return r

    def reset(self) -> dict:
        self.next_waypoint = None

        self.episode += 1
        self.timestep = 0
        self.total_reward = 0.0
        self.past_obs = self._init_past_obs()

        return super().reset()

    def reset_world(self):
        if self.should_sample_town:
            self.set_town(town=random.choice(self.town_set))

        if self.should_sample_weather:
            self.set_weather(weather=random.choice(self.weather_set))

        super().reset_world()

    def reset_info(self):
        for k in self.info_buffer.keys():
            self.info_buffer[k].clear()

    def render(self, *args, **kwargs):
        super().render()
        
        if self.should_record:
            pygame.image.save(self.display, os.path.join(self.record_path, f'{self.timestep}.jpeg'))

    def set_record_path(self, path):
        if isinstance(path, str):
            self.record_path = path
            self.should_record = True
        else:
            self.should_record = False
            self.record_path = None

    def step(self, actions):
        """Performs one environment step (i.e. it updates the world, etc.)"""
        state, reward, done, info = super().step(actions)

        if self.timestep % self.info_every == 0:
            for k, v in info.items():
                self.info_buffer[k].append(v)

        self.timestep += 1
        self.total_reward += reward

        return state, reward, done, info

    def on_collision(self, event: carla.CollisionEvent, **kwargs):
        actor_type = event.other_actor.type_id
        print(f'Collision with actor={actor_type})')
        self.trigger_event(event=CARLAEvent.ON_COLLISION, actor=actor_type)

        if 'pedestrian' in actor_type:
            self.collision_penalty += self.penalty
            self.should_terminate = True

        elif 'vehicle' in actor_type:
            self.collision_penalty += self.penalty / 2.0
            self.should_terminate = True
        else:
            self.collision_penalty += self.penalty / 100.0

        self.should_terminate = True

    def on_sensors_data(self, data: dict) -> dict:
        data = super().on_sensors_data(data)

        if not self.stack_depth:
            return data

        # concatenate depth image along the channel axis
        depth = data['depth']
        r = depth[:, :, 0]
        g = depth[:, :, 1]
        b = depth[:, :, 2]

        depth = (r + g * 256 + b * 256 * 256) / (256**3 - 1)
        depth = np.log1p(depth * 1000.0)

        depth_image = np.concatenate([np.zeros_like(depth), depth, np.zeros_like(depth)], axis=1)
        data['camera'] = np.concatenate((data['camera'], depth_image), axis=-1)
        return data

    def get_observation(self, sensors_data: dict) -> Union[list, dict]:
        obs = self._get_observation(sensors_data)

        # consider an observation (over time) only at certain timesteps
        if self.timestep % self.past_obs_freq == 0:
            # update past observation list:
            self.past_obs.pop(0)  # remove the oldest (t=0)
            self.past_obs.append(obs)  # append the newest

        return self.past_obs.copy()

    def _get_observation(self, sensors_data: dict) -> dict:
        if len(sensors_data.keys()) == 0:
            # return default obs
            return dict(image=self.default_image, vehicle=self.VEHICLE_FEATURES['default'],
                        road=self.ROAD_FEATURES['default'], navigation=self.NAVIGATION_FEATURES['default'])

        # get image, reshape, and scale
        image = np.asarray(sensors_data['camera'], dtype=np.float32)

        if image.shape != self.image_shape:
            image = env_utils.resize(image, size=self.image_size)

        image /= 255.0

        # features
        vehicle_obs = self._get_vehicle_features()
        road_obs = self._get_road_features()
        navigation_obs = self._get_navigation_features()

        obs = dict(image=image, vehicle=vehicle_obs, road=road_obs, navigation=navigation_obs)
        return env_utils.replace_nans(obs)

    def _init_past_obs(self) -> list:
        """Returns a list of empty observations"""
        return [self._get_observation(sensors_data={}) for _ in range(self.time_horizon)]

    def get_info(self) -> dict:
        info = super().get_info()
        info['episode'] = self.episode
        info['timestep'] = self.timestep
        info['total_reward'] = self.total_reward
        info['reward'] = self.reward()
        return info

    def _get_road_features(self):
        """9 features:
            - 3: is_intersection, is_junction, is_at_traffic_light
            - 1: speed_limit
            - 5: traffic_light_state
        """
        waypoint: carla.Waypoint = self.map.get_waypoint(self.vehicle.get_location())
        speed_limit = self.vehicle.get_speed_limit() / 100.0

        # Traffic light:
        is_at_traffic_light = float(self.vehicle.is_at_traffic_light())
        traffic_light_state = self.one_hot_traffic_light_state()

        return np.concatenate((
            [float(waypoint.is_intersection), float(waypoint.is_junction), is_at_traffic_light],
            [speed_limit],
            traffic_light_state), axis=0)

    def _get_vehicle_features(self):
        """4 features:
            - 1: similarity (e.g. current heading direction w.r.t. next route waypoint)
            - 1: speed
            - 1: throttle
            - 1: brake
        """
        return np.array([
            self.similarity,
            carla_utils.speed(self.vehicle) / 100.0,
            self.control.throttle,
            self.control.brake])

    def _get_navigation_features(self):
        """features: N distances from current vehicle location to N next route waypoints' locations
        """
        vehicle_location = self.vehicle.get_location()
        waypoints = self.route.get_next_waypoints(amount=self.num_waypoints)
        distances = []

        for w in waypoints:
            d = carla_utils.l2_norm(vehicle_location, w.transform.location) / self.num_waypoints
            distances.append(d)

        # pad the list with last (thus greater) distance if smaller then required
        if len(distances) < self.num_waypoints:
            for _ in range(self.num_waypoints - len(distances)):
                distances.append(distances[-1])

        return np.array(distances)

    def _update_target_waypoint(self):
        super()._update_target_waypoint()

        if self.next_waypoint is None:
            self.next_waypoint = self.route.next

        elif self.next_waypoint != self.route.next:
            self.next_waypoint = self.route.next

    def one_hot_traffic_light_state(self):
        if self.vehicle.is_at_traffic_light():
            state: carla.TrafficLightState = self.vehicle.get_traffic_light_state()
        else:
            state = carla.TrafficLightState.Unknown

        vector = np.zeros(shape=5, dtype=np.float32)
        vector[state] = 1.0
        return vector

    @staticmethod
    def one_hot_speed(speed: float):
        vector = np.zeros(shape=4, dtype=np.float32)

        if speed <= 30.0:
            vector[0] = 1.0
        elif 30.0 < speed <= 60.0:
            vector[1] = 1.0
        elif 60.0 < speed <= 90.0:
            vector[2] = 1.0
        else:
            # speed > 90.0
            vector[3] = 1.0

        return vector

    @staticmethod
    def one_hot_lane_change(lane: carla.LaneChange):
        vector = np.zeros(shape=4, dtype=np.float32)

        if lane is carla.LaneChange.NONE:
            vector[0] = 1.0
        elif lane is carla.LaneChange.Left:
            vector[1] = 1.0
        elif lane is carla.LaneChange.Right:
            vector[2] = 1.0
        else:
            # lane is carla.LaneChange.Both
            vector[3] = 1.0

        return vector

    @staticmethod
    def one_hot_lane_type(lane: carla.LaneType):
        vector = np.zeros(shape=5, dtype=np.float32)

        if lane is carla.LaneType.NONE:
            vector[0] = 1.0
        elif lane is carla.LaneType.Driving:
            vector[1] = 1.0
        elif lane is carla.LaneType.Sidewalk:
            vector[2] = 1.0
        elif lane is carla.LaneType.Stop:
            vector[3] = 1.0
        else:
            vector[4] = 1.0

        return vector

    @staticmethod
    def one_hot_lane_marking_type(lane: carla.LaneMarkingType):
        vector = np.zeros(shape=4, dtype=np.float32)

        if lane is carla.LaneMarkingType.NONE:
            vector[0] = 1.0
        elif lane is carla.LaneMarkingType.Broken:
            vector[1] = 1.0
        elif lane is carla.LaneMarkingType.Solid:
            vector[2] = 1.0
        else:
            vector[3] = 1.0

        return vector

    def one_hot_similarity(self, threshold=0.3):
        vector = np.zeros(shape=4, dtype=np.float32)

        if self.similarity > 0.0:
            if self.similarity >= 1.0 - threshold:
                vector[0] = 1.0
            else:
                vector[1] = 1.0
        else:
            if self.similarity <= threshold - 1.0:
                vector[2] = 1.0
            else:
                vector[3] = 1.0

        return vector

    @staticmethod
    def one_hot_waypoint_distance(distance: float, very_close=1.5, close=3.0):
        vector = np.zeros(shape=3, dtype=np.float32)

        if distance <= very_close:
            vector[0] = 1.0
        elif very_close < distance <= close:
            vector[1] = 1.0
        else:
            vector[2] = 1.0

        return vector
