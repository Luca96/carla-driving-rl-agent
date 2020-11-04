import carla
import numpy as np

from gym import spaces

from rl import utils
from rl.environments import ThreeCameraCARLAEnvironment, CARLAEvent
from rl.environments.carla.tools import utils as carla_utils

from typing import Dict, Tuple, Optional


class CARLAEnv(ThreeCameraCARLAEnvironment):
    VEHICLE_FEATURES = dict(space=spaces.Box(low=-99.0, high=99.0, shape=(19,)),
                            default=np.zeros(shape=19, dtype=np.float32))

    ROAD_FEATURES = dict(space=spaces.Box(low=0.0, high=1.0, shape=(29,)), default=np.zeros(shape=29, dtype=np.float32))
    COMMAND_SPACE = spaces.Box(low=0.0, high=1.0, shape=(3,))

    def __init__(self, *args, stack_depth=False, collision_penalty=1000.0, info_every=1, add_throttle=0.0,
                 range_controls: Optional[Dict[str, Tuple[float, float]]] = None, **kwargs):
        """
        :param stack_depth: if true the depth-image from the depth camera sensor will be stacked along the channel
                            dimension of the image, resulting in an image with an additional channel (e.g. 3 + 1 = 4)
        :param collision_penalty: how much the agent should be penalized for colliding with other objects.
        :param info_every: how frequently in terms of steps, the additional information should be gathered.
        :param add_throttle: fixed amount of `throttle` added to the vehicle control (i.e. added to `control.throttle`)
        :param range_controls: optional dict used to specify the range for each vehicle's control.
        """
        assert info_every >= 1
        image_shape = kwargs.pop('image_shape', (90, 120, 3))

        if stack_depth:
            self.stack_depth = True
            image_shape = (image_shape[0], image_shape[1], image_shape[2] + 1)
        else:
            self.stack_depth = False

        super().__init__(*args, image_shape=image_shape, **kwargs)

        self.penalty = collision_penalty
        self.next_waypoint = None
        self.waypoint_reward = 0.0
        self.info_every = info_every
        self.add_throttle = add_throttle

        # statistics
        self.episode = -1
        self.timestep = 0
        self.total_reward = 0.0

        self.range_controls = {} if range_controls is None else range_controls
        self.info_buffer = {k: [] for k in self.info_space.spaces.keys()}

    @property
    def info_space(self) -> spaces.Space:
        space: spaces.Dict = super().info_space

        return spaces.Dict(episode=spaces.Discrete(n=1), timestep=spaces.Discrete(n=1),
                           total_reward=spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
                           reward=spaces.Box(low=-np.inf, high=np.inf, shape=(1,)), **space.spaces)

    def actions_to_control(self, actions):
        """Converts the given actions to vehicle's control"""
        super().actions_to_control(actions)

        if carla_utils.speed(self.vehicle) < 10.0:
            self.control.brake = 0.0

        if self.control.throttle > 0.001:
            self.control.throttle = min(self.control.throttle + self.add_throttle, 1.0)

        if 'throttle' in self.range_controls:
            throttle = self.range_controls['throttle']
            self.control.throttle = utils.clip(self.control.throttle, min_value=throttle[0], max_value=throttle[1])

        if 'brake' in self.range_controls:
            brake = self.range_controls['brake']
            self.control.brake = utils.clip(self.control.brake, min_value=brake[0], max_value=brake[1])

        if 'steer' in self.range_controls:
            steer = self.range_controls['steer']
            self.control.steer = utils.clip(self.control.steer, min_value=steer[0], max_value=steer[1])

    def reward(self, *args, s=0.65, d=6.0, **kwargs) -> float:
        """Reward function"""
        speed = carla_utils.speed(self.vehicle)
        dw = self.route.distance_to_next_waypoint()

        if self.collision_penalty > 0.0:
            self.should_terminate = True
            return -self.collision_penalty

        if self.similarity <= s:
            self.should_terminate = True
            self.trigger_event(CARLAEvent.OUT_OF_LANE)
            return -1.0

        if speed > self.vehicle.get_speed_limit() or dw > d:
            return 0.0

        if speed < 1.0:
            return -0.1 * self.similarity

        if 1.0 <= speed <= 2.5:
            return 0.1 * self.similarity

        return self.similarity

    # def reward(self, *args, s=0.65, d=4.0, **kwargs) -> float:
        # """Reward function"""
        # speed = carla_utils.speed(self.vehicle)
        # speed_limit = self.vehicle.get_speed_limit()
        # dw = self.route.distance_to_next_waypoint()
        #
        # if self.collision_penalty > 0.0:
        #     self.should_terminate = True
        #     return -self.collision_penalty
        #
        # if self.similarity <= s:
        #     self.should_terminate = True
        #     self.trigger_event(CARLAEvent.OUT_OF_LANE)
        #     return -1.0
        #
        # if speed > speed_limit or dw > d:
        #     return 0.0
        #
        # # if speed < 1.0:
        # #     return -0.1 * self.similarity
        # #
        # # if 1.0 <= speed <= 2.5:
        # #     return 0.1 * self.similarity
        #
        # return (speed / speed_limit) * self.similarity

    def reset(self) -> dict:
        self.next_waypoint = None
        self.waypoint_reward = 0.0

        self.episode += 1
        self.timestep = 0
        self.total_reward = 0.0

        for k in self.info_buffer.keys():
            self.info_buffer[k].clear()

        return super().reset()

    def step(self, actions):
        """Performs one environment step (i.e. it updates the world, etc.)"""
        state, reward, done, info = super().step(actions)

        if self.timestep % self.info_every == 0:
            for k, v in info.items():
                self.info_buffer[k].append(v)

        self.timestep += 1
        self.total_reward += reward

        return state, reward, done, info

    @staticmethod
    def convert_command(command):
        """Converts the 7-dimensional routing command to a 3-dimensional high-level command to condition the agent"""
        i = np.argmax(command)

        if i == 0:
            # left
            return [1.0, 0.0, 0.0]

        if i == 1:
            # right
            return [0.0, 0.0, 1.0]

        # straight or follow lane
        return [0.0, 1.0, 0.0]

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

    def get_observation(self, sensors_data: dict) -> dict:
        obs = super().get_observation(sensors_data)
        obs['command'] = self.convert_command(obs['command'])
        return obs

    def get_info(self) -> dict:
        info = super().get_info()
        info['episode'] = self.episode
        info['timestep'] = self.timestep
        info['total_reward'] = self.total_reward
        info['reward'] = self.reward()
        return info

    def _get_road_features(self):
        """29 features:
            - 3: is_intersection, is_junction, is_at_traffic_light
            - 4: speed_limit
            - 5: traffic_light_state
            - 4: lane_change
            - 5: lane_type,
            - 8: left + right lane_marking_type
        """
        waypoint: carla.Waypoint = self.map.get_waypoint(self.vehicle.get_location())
        speed_limit_bin = self.one_hot_speed(speed=self.vehicle.get_speed_limit())

        # Traffic light:
        is_at_traffic_light = float(self.vehicle.is_at_traffic_light())
        traffic_light_state = self.one_hot_traffic_light_state()

        # Lanes:
        lane_type = self.one_hot_lane_type(lane=waypoint.lane_type)
        lane_change = self.one_hot_lane_change(lane=waypoint.lane_change)
        left_lane_type = self.one_hot_lane_marking_type(lane=waypoint.left_lane_marking)
        right_lane_type = self.one_hot_lane_marking_type(lane=waypoint.right_lane_marking)

        return np.concatenate((
            [float(waypoint.is_intersection), float(waypoint.is_junction), is_at_traffic_light],
            speed_limit_bin,
            traffic_light_state,
            lane_type,
            lane_change,
            left_lane_type,
            right_lane_type), axis=0)

    def _get_vehicle_features(self):
        """19 features:
            - 4: vehicle's speed
            - 3: accelerometer
            - 3: gyroscope
            - 1: compass
            - 1: similarity
            - 4: similarity (one-hot)
            - 3: distance
        """
        imu_sensor = self.sensors['imu']

        return np.concatenate((
            self.one_hot_speed(speed=carla_utils.speed(self.vehicle)),
            imu_sensor.accelerometer,
            imu_sensor.gyroscope,
            [imu_sensor.compass, self.similarity],
            self.one_hot_similarity(),
            self.one_hot_waypoint_distance(self.route.distance_to_next_waypoint())), axis=0)

    def _update_target_waypoint(self):
        super()._update_target_waypoint()

        if self.next_waypoint is None:
            self.next_waypoint = self.route.next
            self.waypoint_reward = 0.0

        elif self.next_waypoint != self.route.next:
            self.next_waypoint = self.route.next
            self.waypoint_reward = 1.0
        else:
            self.waypoint_reward = 0.0

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
