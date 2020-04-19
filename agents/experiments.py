"""A collection of various experiment settings."""

import math
import carla
import numpy as np

from tensorforce import Agent

from agents import Agents, SensorSpecs, Specs
from agents.environment import SynchronousCARLAEnvironment
from agents import env_utils
from tools import utils


# -------------------------------------------------------------------------------------------------
# -- Baseline Experiments
# -------------------------------------------------------------------------------------------------

class CARLABaselineExperiment(SynchronousCARLAEnvironment):
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

class CARLARouteFollowExperiment(SynchronousCARLAEnvironment):
    """Base class (with basic behaviour) for CARLA Experiments"""

    # skill, throttle/brake intensity, steer
    ACTIONS_SPEC = dict(type='float', shape=(3,), min_value=-1.0, max_value=1.0)
    DEFAULT_ACTIONS = np.array([0.0, 0.0, 0.0])

    # A "skill" is a high-level action
    SKILLS = {0: 'idle',     1: 'brake',
              2: 'forward',  3: 'forward left',  4: 'forward right',
              5: 'backward', 6: 'backward left', 7: 'backward right'}

    # speed, vehicle control (4), accelerometer (3), gyroscope (3), target waypoint's features (5), compass
    VEHICLE_FEATURES_SPEC = dict(type='float', shape=(17,))

    def default_sensors(self) -> dict:
        sensors = super().default_sensors()
        SensorSpecs.set(sensors['camera'], position='on-top2', attachment_type='Rigid')
        return sensors

    def default_agent(self, max_episode_timesteps: int, batch_size=256) -> Agent:
        policy_spec = dict(network=Specs.agent_network_v2(conv=dict(activation='leaky-relu'),
                                                          final=dict(layers=2, units=256)),
                           distributions='gaussian')

        critic_spec = policy_spec.copy()
        critic_spec['temperature'] = 0.5
        critic_spec['optimizer'] = dict(type='synchronization', sync_frequency=1, update_weight=1.0)

        return Agents.ppo_like(self, max_episode_timesteps, policy=policy_spec, critic=critic_spec,
                               batch_size=batch_size,
                               preprocessing=Specs.my_preprocessing(image_shape=(75, 105, 1), stack_images=10),
                               summarizer=Specs.summarizer(frequency=batch_size))

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

    def _get_debug_text(self, actions):
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
                'Collision penalty: %.2f' % self.collision_penalty]


class CARLASegmentationExperiment(CARLARouteFollowExperiment):

    def default_sensors(self) -> dict:
        sensors = super().default_sensors()
        sensors['camera'] = SensorSpecs.segmentation_camera(position='on-top2',
                                                            attachment_type='Rigid',
                                                            image_size_x=self.image_shape[1],
                                                            image_size_y=self.image_shape[0],
                                                            sensor_tick=1.0 / self.fps)
        return sensors

    def on_collision(self, event: carla.CollisionEvent, penalty=1000.0, max_impulse=400.0):
        actor_type = event.other_actor.type_id
        print(f'Collision with {actor_type}')

        if 'pedestrian' in actor_type:
            self.collision_penalty += penalty
            self.should_terminate = True
        elif 'vehicle' in actor_type:
            self.collision_penalty += penalty / 2.0
            self.should_terminate = True
        else:
            self.collision_penalty += penalty / 10.0
            self.should_terminate = False


class CARLAActionPenaltyExperiment(CARLARouteFollowExperiment):
    pass


class CARLAPlayEnvironment(CARLARouteFollowExperiment):
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

    def default_agent(self) -> Agent:
        return Agents.dummy.keyboard(self)

    def actions_to_control(self, actions):
        self.control.throttle = actions[0]
        self.control.steer = actions[1]
        self.control.brake = actions[2]
        self.control.reverse = bool(actions[3])
        self.control.hand_brake = bool(actions[4])

    def on_pre_world_step(self):
        if self.should_debug:
            self.route.draw_route(self.world.debug, life_time=1.0 / self.fps)
            self.route.draw_next_waypoint(self.world.debug, self.vehicle.get_location(), life_time=1.0 / self.fps)


# -------------------------------------------------------------------------------------------------
# -- Pretraining Experiments
# -------------------------------------------------------------------------------------------------

# TODO: improve, solve the issue with env.reset()
class CARLAPretrainExperiment(CARLARouteFollowExperiment):

    def default_sensors(self) -> dict:
        return super().default_sensors()

    def default_agent(self) -> Agent:
        raise NotImplementedError

    def reward(self, actions, time_cost=-1.0, b=-1000.0, c=2.0, d=2.0):
        speed = utils.speed(self.vehicle)
        direction_penalty = speed + 1
        efficiency_term = 0.0

        speed_limit = self.vehicle.get_speed_limit()

        if speed <= speed_limit:
            speed_penalty = 0.0 if speed > 10.0 else -1.0
        else:
            speed_penalty = c * (speed_limit - speed)

        return time_cost - self.collision_penalty + efficiency_term + direction_penalty + speed_penalty

    @staticmethod
    def _skill_from_control(control: carla.VehicleControl) -> (float, str):
        throttle = control.throttle
        steer = control.steer
        brake = control.brake
        reverse = control.reverse

        if throttle == 0.0 and brake > 0.0:
            return -0.7, 'brake'
        elif not reverse:
            if (throttle > 0.0) and (brake == 0.0) and (steer == 0.0):
                return -0.4, 'forward'
            elif (throttle > 0.0) and (brake == 0.0) and (steer < 0.0):
                return -0.15, 'forward left'
            elif (throttle > 0.0) and (brake == 0.0) and (steer > 0.0):
                return 0.1, 'forward right'
            else:
                return -1.0, 'idle'
        else:
            if (throttle > 0.0) and (brake == 0.0) and (steer == 0.0):
                return 0.4, 'backward'
            elif (throttle > 0.0) and (brake == 0.0) and (steer > 0.0):
                return 0.7, 'backward left'
            elif (throttle > 0.0) and (brake == 0.0) and (steer < 0.0):
                return 0.9, 'backward right'

    def control_to_actions(self, control: carla.VehicleControl):
        skill, name = self._skill_from_control(control)

        if control.throttle > 0.0:
            return [skill, control.throttle, control.steer], name
        else:
            return [skill, control.brake, control.steer], name


# -------------------------------------------------------------------------------------------------
# -- Curriculum Learning Experiment
# -------------------------------------------------------------------------------------------------

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
