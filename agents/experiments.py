"""A collection of various experiment settings."""

import cv2
import math
import carla
import pygame
import numpy as np

from typing import Optional, ClassVar, List, Union
from tensorforce import Agent

from agents.agents import Agents
from agents.specifications import Specifications as Specs
from agents.environment import SynchronousCARLAEnvironment
from agents import env_utils
from agents.sensors import SensorSpecs
from agents.env_features import ActionPenalty, TemporalFeature
from agents.specifications import NetworkSpec
from tools import utils
from tools.utils import WAYPOINT_DICT


# -------------------------------------------------------------------------------------------------
# -- Baseline Experiments
# -------------------------------------------------------------------------------------------------


class BaselineExperiment(SynchronousCARLAEnvironment):
    ACTIONS_SPEC = dict(type='float', shape=(3,), min_value=-1.0, max_value=1.0)
    DEFAULT_ACTIONS = np.array([0., 0., 0.])

    # vehicle: speed, accelerometer (x, y, z), gyroscope (x, y, z), position (x, y), destination (x, y), compass
    VEHICLE_FEATURES_SPEC = dict(type='float', shape=(12,))

    def default_sensors(self) -> dict:
        return dict(imu=SensorSpecs.imu(),
                    collision=SensorSpecs.collision_detector(),
                    camera=SensorSpecs.rgb_camera(position='front',
                                                  attachment_type='Rigid',
                                                  image_size_x=self.window_size[0], image_size_y=self.window_size[1],
                                                  sensor_tick=1.0 / self.fps))

    def default_agent(self, **kwargs) -> Agent:
        return Agents.baseline(self, **kwargs)

    def reward(self, actions, time_cost=-1.0, b=-1000.0, c=2.0, d=2.0):
        # Direction term: alignment of the vehicle's heading direction with the waypoint's forward vector
        closest_waypoint = self.route.next.waypoint
        similarity = utils.cosine_similarity(self.vehicle.get_transform().get_forward_vector(),  # heading direction
                                             closest_waypoint.transform.get_forward_vector())
        speed = utils.speed(self.vehicle)

        if similarity > 0:
            direction_penalty = (speed + 1) * similarity  # speed + 1, to avoid 0 speed
        else:
            direction_penalty = (speed + 1) * similarity * d

        if self.travelled_distance <= self.route.size:
            efficiency_term = 0.0
        else:
            efficiency_term = -(self.travelled_distance - self.route.size) - self.route.distance_to_destination()

            # travelled more than route size, zero direction_penalty if positive
            if direction_penalty > 0.0:
                direction_penalty = 0.0

        # Speed-limit compliance:
        speed_limit = self.vehicle.get_speed_limit()

        if speed <= speed_limit:
            speed_penalty = 0.0 if speed > 10.0 else -1.0
        else:
            speed_penalty = c * (speed_limit - speed)

        return time_cost - self.collision_penalty + efficiency_term + direction_penalty + speed_penalty

    def _get_vehicle_features(self):
        t = self.vehicle.get_transform()

        imu_sensor = self.sensors['imu']
        gyroscope = imu_sensor.gyroscope
        accelerometer = imu_sensor.accelerometer

        return [min(utils.speed(self.vehicle), 150.0),
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
                # Destination:
                self.destination.x,
                self.destination.y,
                # Compass:
                math.radians(imu_sensor.compass)]

    def actions_to_control(self, actions):
        # Throttle
        if actions[0] < 0:
            self.control.throttle = 0.0
            self.control.brake = 1.0
        else:
            self.control.throttle = 1.0
            self.control.brake = 0.0

        # if actions[0] < -0.33:
        #     self.control.throttle = 0.3
        # elif actions[0] > 0.33:
        #     self.control.throttle = 0.9
        # else:
        #     self.control.throttle = 0.5

        # Steer
        if actions[1] < -0.33:
            self.control.steer = -0.5
        elif actions[1] > 0.33:
            self.control.steer = 0.5
        else:
            self.control.steer = 0

        # self.control.reverse = bool(actions[2] < 0)
        self.control.brake = 0.0 if actions[2] < 0 else float(actions[2])


# -------------------------------------------------------------------------------------------------
# -- Experiments
# -------------------------------------------------------------------------------------------------

class RouteFollowExperiment(SynchronousCARLAEnvironment):
    """Base class (with basic behaviour) for CARLA Experiments"""

    # skill, throttle/brake intensity, steer
    ACTIONS_SPEC = dict(type='float', shape=(3,), min_value=-1.0, max_value=1.0)
    DEFAULT_ACTIONS = np.array([0.0, 0.0, 0.0])

    # A "skill" is a high-level action
    SKILLS = {0: 'idle', 1: 'brake',
              2: 'forward', 3: 'forward left', 4: 'forward right',
              5: 'backward', 6: 'backward left', 7: 'backward right'}

    # speed, vehicle control (4), accelerometer (3), gyroscope (3), target waypoint's features (5), compass
    VEHICLE_FEATURES_SPEC = dict(type='float', shape=(17,))

    def default_sensors(self) -> dict:
        sensors = super().default_sensors()
        SensorSpecs.set(sensors['camera'], position='on-top2', attachment_type='Rigid')
        return sensors

    def default_agent(self, max_episode_timesteps: int, batch_size=256) -> Agent:
        policy_spec = dict(network=Specs.network_v2(conv=dict(activation='leaky-relu'),
                                                    final=dict(layers=2, units=256)),
                           distributions='gaussian')

        critic_spec = policy_spec.copy()
        critic_spec['temperature'] = 0.5
        critic_spec['optimizer'] = dict(type='synchronization', sync_frequency=1, update_weight=1.0)

        return Agents.ppo_like(self, max_episode_timesteps, policy=policy_spec, critic=critic_spec,
                               batch_size=batch_size,
                               preprocessing=Specs.my_preprocessing(image_shape=(75, 105, 1), stack_images=10),
                               summarizer=Specs.summarizer(frequency=batch_size))

    def terminal_condition(self, distance_threshold=2.0):
        super().terminal_condition(distance_threshold=distance_threshold)

    def get_skill_name(self):
        skill = env_utils.scale(self.prev_actions[0])
        return self.SKILLS[int(skill)]

    def actions_to_control(self, actions):
        skill = self.get_skill_name()
        reverse = self.control.reverse

        if skill == 'brake':
            throttle = 0.0
            brake = max(0.1, (actions[1] + 1) / 2.0)
            steer = float(actions[2])
        elif skill == 'forward':
            throttle = max(0.1, (actions[1] + 1) / 2.0)
            brake = 0.0
            steer = 0.0
            reverse = False
        elif skill == 'forward right':
            throttle = max(0.1, (actions[1] + 1) / 2.0)
            brake = 0.0
            steer = max(0.1, abs(actions[2]))
            reverse = False
        elif skill == 'forward left':
            throttle = max(0.1, (actions[1] + 1) / 2.0)
            brake = 0.0
            steer = min(-0.1, -abs(actions[2]))
            reverse = False
        elif skill == 'backward':
            throttle = max(0.1, (actions[1] + 1) / 2.0)
            brake = 0.0
            steer = 0.0
            reverse = True
        elif skill == 'backward left':
            throttle = max(0.1, (actions[1] + 1) / 2.0)
            brake = 0.0
            steer = max(0.1, abs(actions[2]))
            reverse = True
        elif skill == 'backward right':
            throttle = max(0.1, (actions[1] + 1) / 2.0)
            brake = 0.0
            steer = min(-0.1, -abs(actions[2]))
            reverse = True
        else:
            # idle/stop
            throttle = 0.0
            brake = 0.0
            steer = 0.0
            reverse = False

        self.control.throttle = float(throttle)
        self.control.brake = float(brake)
        self.control.steer = float(steer)
        self.control.reverse = reverse
        self.control.hand_brake = False

    def _get_vehicle_features(self):
        control = self.vehicle.get_control()
        imu_sensor = self.sensors['imu']
        gyroscope = imu_sensor.gyroscope
        accelerometer = imu_sensor.accelerometer

        # TODO: substitute accelerometer with vehicle.get_acceleration()? (3D vector)
        # TODO: consider adding 'vehicle.get_angular_velocity()' (3D vector)
        # TODO: substitute speed with 'vehicle.get_velocity()'? (3D vector)
        # TODO: add vehicle's light state

        return [math.log2(1.0 + utils.speed(self.vehicle)),  # speed
                # Vehicle control:
                control.throttle,
                control.steer,
                control.brake,
                float(control.reverse),
                # Accelerometer:
                accelerometer[0],
                accelerometer[1],
                accelerometer[2],
                # Gyroscope:
                gyroscope[0],
                gyroscope[1],
                gyroscope[2],
                # Target (next) waypoint's features:
                self.similarity,
                self.forward_vector.x,
                self.forward_vector.y,
                self.forward_vector.z,
                self.route.distance_to_next_waypoint(),
                # Compass:
                math.radians(imu_sensor.compass)]

    def debug_text(self, actions):
        speed_limit = self.vehicle.get_speed_limit()
        speed = utils.speed(self.vehicle)

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
                'Hand brake: %s' % ('T' if self.control.hand_brake else 'F'),
                '',
                speed_text,
                'Speed limit %.1f km/h' % speed_limit,
                '',
                'Similarity %.2f' % self.similarity,
                'Waypoint\'s Distance %.2f' % self.route.distance_to_next_waypoint(),
                '',
                'Reward: %.2f' % self.reward(actions),
                'Collision penalty: %.2f' % self.collision_penalty,
                'Skill: %s' % self.get_skill_name()]


class RadarSegmentationExperiment(RouteFollowExperiment):
    """Equips the vehicle with RADAR and semantic segmentation camera"""

    def default_sensors(self) -> dict:
        sensors = super().default_sensors()
        sensors['camera'] = SensorSpecs.segmentation_camera(position='on-top2',
                                                            attachment_type='Rigid',
                                                            image_size_x=self.image_size[0],
                                                            image_size_y=self.image_size[1],
                                                            sensor_tick=self.tick_time)
        # sensors['depth'] = SensorSpecs.depth_camera(position='on-top2',
        #                                             attachment_type='Rigid',
        #                                             image_size_x=self.image_size[0],
        #                                             image_size_y=self.image_size[1],
        #                                             sensor_tick=self.tick_time)

        sensors['radar'] = SensorSpecs.radar(position='radar', sensor_tick=self.tick_time)
        return sensors

    def on_collision(self, event: carla.CollisionEvent, penalty=1000.0, max_impulse=400.0):
        actor_type = event.other_actor.type_id

        if 'pedestrian' in actor_type:
            self.collision_penalty += penalty
            self.should_terminate = True
        elif 'vehicle' in actor_type:
            self.collision_penalty += penalty / 2.0
            self.should_terminate = True
        else:
            self.collision_penalty += penalty / 10.0
            self.should_terminate = False

    # def on_sensors_data(self, data: dict) -> dict:
    #     data = super().on_sensors_data(data)
    #     data['depth'] = self.sensors['depth'].convert_image(data['depth'])
    #     depth = env_utils.cv2_grayscale(data['depth'], depth=3)
    #     # data['camera'] = cv2.multiply(data['camera'], cv2.log(1.0 * data['depth']))
    #
    #     data['camera'] = np.multiply(data['camera'], 255 - depth)
    #
    #     # data['camera'] = env_utils.cv2_grayscale(data['camera'], depth=3)
    #     return data

    def render(self, sensors_data: dict):
        super().render(sensors_data)
        print('points:', sensors_data['radar'].get_detection_count())
        env_utils.draw_radar_measurement(debug_helper=self.world.debug, data=sensors_data['radar'])


class CompleteStateExperiment(RouteFollowExperiment):
    """Equips sensors: semantic camera + depth camera + radar"""

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

    def __init__(self, time_horizon=10, radar_shape=(50, 40, 1), *args, **kwargs):
        assert isinstance(radar_shape, tuple)
        super().__init__(*args, **kwargs)
        self.radar_shape = radar_shape

        # TODO: try dtype=np.float16 to save memory
        self.vehicle_obs = TemporalFeature(time_horizon, shape=self.VEHICLE_FEATURES_SPEC['shape'])
        self.skills_obs = TemporalFeature(time_horizon, shape=self.SKILL_SPEC['shape'])
        self.actions_obs = TemporalFeature(time_horizon, shape=self.DEFAULT_CONTROL.shape)
        self.radar_obs = TemporalFeature(time_horizon * 50, shape=(4,))
        self.image_obs = TemporalFeature(time_horizon, shape=self.image_shape[:2], axis=-1)
        self.road_obs = TemporalFeature(time_horizon, shape=self.ROAD_FEATURES_SPEC['shape'])

    def default_sensors(self) -> dict:
        sensors = super().default_sensors()
        sensors['camera'] = SensorSpecs.segmentation_camera(position='on-top2', attachment_type='Rigid',
                                                            image_size_x=self.image_size[0],
                                                            image_size_y=self.image_size[1],
                                                            sensor_tick=self.tick_time)
        sensors['radar'] = SensorSpecs.radar(position='radar', sensor_tick=self.tick_time)
        return sensors

    def states(self):
        return dict(image=dict(shape=self.image_obs.shape),
                    radar=dict(type='float', shape=self.radar_obs.shape),
                    road=dict(type='float', shape=self.road_obs.shape),
                    vehicle=dict(type='float', shape=self.vehicle_obs.shape),
                    past_actions=dict(type='float', shape=self.actions_obs.shape),
                    past_skills=dict(type='float', shape=self.skills_obs.shape, min_value=0.0,
                                     max_value=len(self.SKILLS) - 1.0))

    def actions(self):
        return dict(control=self.CONTROL_SPEC, skill=self.SKILL_SPEC)

    def policy_network(self, **kwargs) -> List[dict]:
        features = dict(road=dict(shape=self.road_obs.shape, filters=6, kernel=3, stride=1, layers=4),
                        vehicle=dict(shape=self.vehicle_obs.shape, filters=6, kernel=(3, 4), layers=4),
                        past_actions=dict(shape=self.actions_obs.shape, filters=6, kernel=(3, 1), layers=4))

        conv_nets = dict(image=dict(filters=22, layers=(2, 5), middle_noise=True, middle_normalization=True),
                         radar=dict(filters=12, reshape=self.radar_shape, layers=(2, 2), activation1='elu', noise=0.0))

        dense_nets = dict(past_skills=dict(units=[24, 30, 30, 30, 24], activation='swish'))  # 24 -> ~3.6k

        # < 0.02ms (agent.act)
        return Specs.network_v4(convolutional=conv_nets, features=features, dense=dense_nets,
                                final=dict(units=[320, 224, 224, 128], activation='swish'))  # 284 -> ~242k

    def reward(self, actions, time_cost=-1.0, b=2.0, c=2.0, d=2.0, k=6.0):
        # normalize reward to [-k, +1] where 'k' is an arbitrary multiplier representing a big negative value
        v = max(utils.speed(self.vehicle), 1.0)
        r = super().reward(actions)
        r = max(r, -k * v)
        return (r / v) + self.action_penalty(actions)

    def reset(self, soft=False) -> dict:
        # reset observations (np.copyto() should reuse memory...)
        self.actions_obs.reset()
        self.radar_obs.reset()
        self.image_obs.reset()
        self.road_obs.reset()
        self.skills_obs.reset()
        self.vehicle_obs.reset()

        return super().reset(soft=soft)

    def actions_to_control(self, actions):
        """Specifies the mapping between an actions vector and the vehicle's control."""
        actions = actions['control']
        self.control.throttle = float(actions[0]) if actions[0] > 0 else 0.0
        self.control.brake = float(-actions[0]) if actions[0] < 0 else 0.0
        self.control.steer = float(actions[1])
        self.control.reverse = bool(actions[2] > 0)
        self.control.hand_brake = False

    def get_skill_name(self):
        """Returns skill's name"""
        index = round(self.prev_actions['skill'][0])
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

    # def render(self, sensors_data: dict):
    #     # depth = camera.convert(data)
    #     # depth = np.stack((depth,) * 3, axis=-1) / depth.max() * 255.0
    #     # print(depth.shape, depth.min(), depth.max())
    #
    #     # sensors_data['camera'] = np.mean(sensors_data['camera'], axis=-1)
    #     # sensors_data['camera'] = env_utils.to_grayscale(sensors_data['camera'])
    #     # sensors_data['camera'] = sensors_data['camera'][..., ::-1]
    #     super().render(sensors_data)

    def on_sensors_data(self, data: dict) -> dict:
        data = super().on_sensors_data(data)
        data['radar'] = self.sensors['radar'].convert(data['radar'])
        return data

    def debug_text(self, actions):
        text = super().debug_text(actions)
        text[-1] = 'Skill (%d) = %s' % (round(self.prev_actions['skill'][0]), self.get_skill_name())
        text.append('Coordination %d' % self.action_penalty(actions))

        return text

    def _get_observation(self, sensors_data: dict) -> dict:
        if len(sensors_data.keys()) == 0:
            # sensor_data is empty so, return a default observation
            return dict(image=self.image_obs.default, radar=self.radar_obs.default, vehicle=self.vehicle_obs.default,
                        road=self.road_obs.default, past_actions=self.actions_obs.default,
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
        self.actions_obs.append(value=self.prev_actions['control'].copy())
        self.skills_obs.append(value=self.prev_actions['skill'].copy())
        self.road_obs.append(value=self._get_road_features())
        self.image_obs.append(image, depth=True)

        # copy radar measurements
        for i, detection in enumerate(radar):
            self.radar_obs.append(detection)

        # observation
        return dict(image=self.image_obs.data, radar=self.radar_obs.data, vehicle=self.vehicle_obs.data,
                    road=self.road_obs.data, past_actions=self.actions_obs.data, past_skills=self.skills_obs.data)


class SkipTrickExperiment(CompleteStateExperiment):
    # TODO: take the average/maximum/minimum reward when skipping?
    # tells how many times the current action (control) should be repeated
    SKIP_SPEC = dict(type='float', shape=1, min_value=1.0, max_value=6.0)  # 6 means repeat for 200ms
    DEFAULT_SKIP = 0.0

    DEFAULT_CONTROL = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    DEFAULT_SKILL = np.array([0.0], dtype=np.float32)

    DEFAULT_ACTIONS = dict(control=DEFAULT_CONTROL, skill=DEFAULT_SKILL, skip=DEFAULT_SKIP)

    # Vehicle: speed, acceleration, angular velocity, similarity, distance to waypoint
    VEHICLE_FEATURES_SPEC = dict(type='float', shape=(5,))

    # Road: intersection (bool), junction (bool), speed_limit, traffic_light (presence + state), lane type and change,
    ROAD_FEATURES_SPEC = dict(type='float', shape=(7,))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        del self.EMPTY_ACTIONS
        del self.actions_obs

        self.DEFAULT_CONTROLS = np.zeros((self.time_horizon, 4), dtype=np.float32)
        self.DEFAULT_SKIPS = np.zeros((self.time_horizon,), dtype=np.float32)

        # empty observations (to be filled on temporal axis)
        # self.vehicle_obs = self.DEFAULT_VEHICLE.copy()
        self.control_obs = self.DEFAULT_CONTROLS.copy()
        # self.skills_obs = self.DEFAULT_SKILLS.copy()
        self.skips_obs = self.DEFAULT_SKIPS.copy()
        # self.radar_obs = self.DEFAULT_RADAR.copy()
        # self.image_obs = self.DEFAULT_IMAGE.copy()
        # self.road_obs = self.DEFAULT_ROAD.copy()

        self.prev_control = None

    def states(self):
        return dict(image=dict(shape=self.image_shape),
                    radar=dict(type='float', shape=self.DEFAULT_RADAR.shape),
                    road=dict(type='float', shape=self.DEFAULT_ROAD.shape),
                    vehicle=dict(type='float', shape=self.DEFAULT_VEHICLE.shape),
                    past_control=dict(type='float', shape=self.DEFAULT_CONTROLS.shape),
                    skills=dict(type='float', shape=self.DEFAULT_SKILLS.shape, min_value=0.0,
                                max_value=len(self.SKILLS) - 1.0),
                    action_skip=dict(type='float', shape=self.DEFAULT_SKIPS.shape, min_value=1.0, max_value=6.0))

    def actions(self):
        return dict(control=self.CONTROL_SPEC, skill=self.SKILL_SPEC, skip=self.SKIP_SPEC)

    def on_collision(self, event: carla.CollisionEvent, penalty=1000.0, **kwargs):
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

    def reward(self, actions, time_cost=-1, d=2.0, w=10.0, s=2.0, v_max=150.0, d_max=100.0, **kwargs):
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

    def execute(self, actions, record_path: str = None):
        repeat_actions = int(np.round(actions['skip']))
        state = None
        terminal = False
        avg_reward = 0.0

        for _ in range(repeat_actions):
            state, terminal, reward = super().execute(actions, record_path)
            avg_reward += reward

            if terminal:
                break

        return state, terminal, avg_reward / repeat_actions

    def reset(self, soft=False) -> dict:
        self.time_index = 0
        self.radar_index = 0
        self.prev_control = [0.0, 0.0, 0.0, 0.0]

        # reset observations
        np.copyto(dst=self.control_obs, src=self.DEFAULT_CONTROLS)
        np.copyto(dst=self.road_obs, src=self.DEFAULT_ROAD)
        np.copyto(dst=self.radar_obs, src=self.DEFAULT_RADAR)
        np.copyto(dst=self.image_obs, src=self.DEFAULT_IMAGE)
        np.copyto(dst=self.skills_obs, src=self.DEFAULT_SKILLS)
        np.copyto(dst=self.skips_obs, src=self.DEFAULT_SKIPS)
        np.copyto(dst=self.vehicle_obs, src=self.DEFAULT_VEHICLE)

        return RouteFollowExperiment.reset(self, soft=soft)

    def debug_text(self, actions):
        text = super().debug_text(actions)
        text.append('Skip: %d' % int(np.round(actions['skip'])))
        return text

    def _get_observation(self, sensors_data: dict) -> dict:
        if len(sensors_data.keys()) == 0:
            # sensor_data is empty so, return a default observation
            return dict(image=self.DEFAULT_IMAGE, radar=self.DEFAULT_RADAR, vehicle=self.DEFAULT_VEHICLE,
                        road=self.DEFAULT_ROAD, past_control=self.DEFAULT_CONTROLS, skills=self.DEFAULT_SKILLS,
                        action_skip=self.DEFAULT_SKIPS)

        # resize image if necessary
        image = sensors_data['camera']

        if image.shape != self.image_shape:
            image = env_utils.resize(image, size=self.image_size)

        # grayscale image, and rescale to [-1, +1]
        image = (2 * env_utils.cv2_grayscale(image) - 255.0) / 255.0
        radar = sensors_data['radar']
        t = self.time_index

        # concat new observations along the temporal axis:
        self.vehicle_obs[t] = self._get_vehicle_features()
        self.control_obs[t] = self.prev_control.copy()
        self.skills_obs[t] = self.prev_actions['skill'].copy()
        self.skips_obs[t] = self.prev_actions['skip'].copy()
        self.road_obs[t] = np.array(self._get_road_features(), dtype=np.float32)
        self.image_obs[:, :, t] = image

        # copy radar measurements
        for i, detection in enumerate(radar):
            index = (self.radar_index + i) % self.radar_obs.shape[0]
            self.radar_obs[index] = detection

        # observation
        return dict(image=self.image_obs, radar=self.radar_obs, vehicle=self.vehicle_obs, road=self.road_obs,
                    past_control=self.control_obs, skills=self.skills_obs, action_skip=self.skips_obs)

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

    @staticmethod
    def run(num_episodes: int, num_timesteps: int, env_args=dict(), agent_args=dict()):
        radar_shape = env_args.pop('radar_shape')
        env = SkipTrickExperiment(debug=True, **env_args)

        # agent network
        features = dict(road=dict(shape=env.DEFAULT_ROAD.shape, filters=6//2, kernel=3, stride=1, layers=4),
                        vehicle=dict(shape=env.DEFAULT_VEHICLE.shape, filters=6//2, kernel=(3, 4), layers=4),
                        past_control=dict(shape=env.DEFAULT_CONTROLS.shape, filters=6//2, kernel=(3, 1), layers=4))

        conv_nets = dict(image=dict(filters=20//5, layers=(2, 5), middle_noise=True, middle_normalization=True),
                         radar=dict(filters=10//2, reshape=radar_shape + (1,), layers=(2//2, 2//2), activation1='elu',
                                    noise=0.0))

        dense_nets = dict(skills=dict(units=[30//3, 30//3], activation='swish'),
                          action_skip=dict(units=[24//4, 24//4], activation='swish'))

        network = Specs.network_v4(convolutional=conv_nets, features=features, dense=dense_nets,
                                   final=dict(units=[256//8, 128//8], activation='swish'))

        agent = Agents.ppo6(env, max_episode_timesteps=num_timesteps, network=network, summarizer=Specs.summarizer(),
                            **agent_args)

        # fitting
        env.train(agent, num_episodes, num_timesteps, weights_dir='weights/ppo6', agent_name='ppo6',
                  record_dir=None)


class RouteCommandExperiment(RouteFollowExperiment):
    pass


# -------------------------------------------------------------------------------------------------
# -- Play Environments
# -------------------------------------------------------------------------------------------------

# TODO: override 'train' (if necessary) -> consider to add 'play' and 'record' methods instead of 'train'
class CARLAPlayEnvironment(RouteFollowExperiment):
    ACTIONS_SPEC = dict(type='float', shape=(5,), min_value=-1.0, max_value=1.0)
    DEFAULT_ACTIONS = [0.0, 0.0, 0.0, 0.0, 0.0]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('Controls: (W, or UP) accelerate, (A or LEFT) steer left, (D or RIGHT) steer right, (S or DOWN) brake, '
              '(Q) toggle reverse, (SPACE) hand-brake, (ESC) quit.')

    def default_sensors(self) -> dict:
        sensors = super().default_sensors()
        sensors['camera']['transform'] = SensorSpecs.get_position('top')
        return sensors

    def default_agent(self, **kwargs) -> Agent:
        return Agents.keyboard(self)

    def play(self):
        """Let's you control the vehicle with a keyboard."""
        states = self.reset()
        agent = self.default_agent()
        terminal = False

        try:
            with self.synchronous_context:
                while not terminal:
                    actions = agent.act(states)
                    states, terminal, reward = self.execute(actions)
                    agent.observe(reward, terminal)
        finally:
            agent.close()
            self.close()

    def actions_to_control(self, actions):
        self.control.throttle = actions[0]
        self.control.steer = actions[1]
        self.control.brake = actions[2]
        self.control.reverse = bool(actions[3])
        self.control.hand_brake = bool(actions[4])

    def before_world_step(self):
        if self.should_debug:
            self.route.draw_route(self.world.debug, life_time=1.0 / self.fps)
            self.route.draw_next_waypoint(self.world.debug, self.vehicle.get_location(), life_time=1.0 / self.fps)


class PlayEnvironment2(RadarSegmentationExperiment, CARLAPlayEnvironment):

    def before_world_step(self):
        pass


class PlayEnvironment3(CompleteStateExperiment, CARLAPlayEnvironment):

    def before_world_step(self):
        pass


# -------------------------------------------------------------------------------------------------
# -- Pretraining Experiments
# -------------------------------------------------------------------------------------------------

class CARLACollectExperience(CompleteStateExperiment):

    def default_agent(self, **kwargs) -> Agent:
        return Agents.pretraining(self, speed=30.0, **kwargs)

    def reward(self, actions, time_cost=-1.0, b=2.0, c=2.0, d=2.0, k=6.0):
        speed = utils.speed(self.vehicle)
        direction_penalty = speed + 1
        speed_limit = self.vehicle.get_speed_limit()

        if speed <= speed_limit:
            speed_penalty = 0.0 if speed > 10.0 else -1.0
        else:
            speed_penalty = c * (speed_limit - speed)

        r = time_cost - self.collision_penalty + direction_penalty + speed_penalty

        # normalize
        v = max(speed, 1.0)
        r = max(r, -k * v)
        return (r / v) + self.action_penalty(actions)

    @staticmethod
    def action_penalty(actions, eps=0.05) -> float:
        ap = CompleteStateExperiment.action_penalty(actions)
        assert ap == len(actions['control'])
        return ap

    def _skill_from_control(self, control: carla.VehicleControl, eps=0.05) -> (float, str):
        t = control.throttle
        s = control.steer
        b = control.brake
        r = control.reverse

        # backward:
        if r and (t > eps) and (b <= eps):
            if s > eps:
                skill = 9
            elif s < -eps:
                skill = 8
            else:
                skill = 7
        # forward:
        elif (not r) and (t > eps) and (b <= eps):
            if s > eps:
                skill = 6
            elif s < -eps:
                skill = 5
            else:
                skill = 4
        # steer:
        elif (t <= eps) and (b <= eps):
            if s > eps:
                skill = 2
            elif s < -eps:
                skill = 3
            else:
                skill = 0
        # brake:
        elif b > eps:
            skill = 1
        else:
            skill = 0

        return skill, self.SKILLS[skill]

    def control_to_actions(self, control: carla.VehicleControl):
        skill, name = self._skill_from_control(control)
        skill = np.array([skill], dtype=np.float32)
        steer = control.steer
        reverse = bool(control.reverse > 0)

        if control.throttle > 0.0:
            return dict(control=[control.throttle, steer, reverse], skill=skill), name
        else:
            return dict(control=[-control.brake, steer, reverse], skill=skill), name

    def debug_text(self, actions):
        text = super().debug_text(actions)
        return text[:11] + text[14:]


# -------------------------------------------------------------------------------------------------
# -- Curriculum Learning Experiment
# -------------------------------------------------------------------------------------------------

class CurriculumLearning:
    pass

# TODO: review implementation
# class CurriculumCARLAEnvironment(SynchronousCARLAEnvironment):
#
#     def learn(self, agent: Agent, initial_timesteps: int, difficulty=1, increment=5, num_stages=1, max_timesteps=1024,
#               trials_per_stage=5, max_repetitions=1, save_agent=True, load_agent=False, agent_name='carla-agent'):
#         initial_difficulty = difficulty
#         target_difficulty = initial_difficulty + num_stages * increment
#
#         if load_agent:
#             agent.load(directory='weights/agents', filename=agent_name, environment=self)
#             print('Agent loaded.')
#
#         for difficulty in range(initial_difficulty, target_difficulty + 1, increment):
#             for r in range(max_repetitions):
#                 success_rate, avg_reward = self.stage(agent,
#                                                       trials=trials_per_stage,
#                                                       difficulty=difficulty,
#                                                       max_timesteps=min(initial_timesteps * difficulty, max_timesteps))
#
#                 print(f'[D-{difficulty}] success_rate: {round(success_rate, 2)}, avg_reward: {round(avg_reward, 2)}')
#
#                 if save_agent:
#                     agent.save(directory='weights/agents', filename=agent_name)
#                     print(f'[D-{difficulty}] Agent saved.')
#
#                 print(f'Repetition {r}-D-{difficulty} ended.')
#
#     def stage(self, agent: Agent, trials: int, difficulty: int, max_timesteps: int):
#         assert trials > 0
#         assert difficulty > 0
#         assert max_timesteps > 0
#
#         # self.reset(soft=False, route_size=difficulty)
#         avg_reward = 0.0
#         success_count = 0
#
#         for trial in range(trials):
#             # states = self.reset(soft=trial != 0, route_size=difficulty)
#             states = self.reset(route_size=difficulty)
#             trial_reward = 0.0
#
#             with self.synchronous_context:
#                 for t in range(max_timesteps):
#                     actions = agent.act(states)
#                     states, terminal, reward = self.execute(actions, distance_threshold=3.0)
#
#                     trial_reward += reward
#                     terminal = terminal or (t == max_timesteps - 1)
#
#                     if self.is_at_destination():
#                         agent.observe(reward, terminal=True)
#                         success_count += 1
#                         print(f'[T-{trial}] Successful -> reward: {round(trial_reward, 2)}')
#                         break
#
#                     elif terminal:
#                         agent.observe(reward, terminal=True)
#                         print(f'[T-{trial}] not successful -> reward: {round(trial_reward, 2)}')
#                         break
#                     else:
#                         agent.observe(reward, terminal=False)
#
#             avg_reward += trial_reward
#
#         return success_count / trials, avg_reward / trials
#
#     def is_at_destination(self, distance_threshold=2.0):
#         return self.route.distance_to_destination() < distance_threshold
#
#     def _get_observation(self, image):
#         if image is None:
#             image = np.zeros(shape=self.image_shape, dtype=np.uint8)
#
#         if image.shape != self.image_shape:
#             image = env_utils.resize(image, size=self.image_size)
#
#         return dict(image=image / 255.0,
#                     vehicle_features=self._get_vehicle_features(),
#                     road_features=self._get_road_features(),
#                     previous_actions=self.prev_actions)
