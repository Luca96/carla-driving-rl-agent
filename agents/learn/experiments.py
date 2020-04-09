"""A collection of various experiment settings."""

import os
import math
import pygame
import carla
import numpy as np
from tensorforce import Agent

from agents.learn import SynchronousCARLAEnvironment, SensorSpecs, env_utils
from agents.specifications import Specifications as Specs
from worlds import utils


# TODO: revise method's arguments
class CARLAExperiment(SynchronousCARLAEnvironment):
    """Base class for running experiments on CARLA simulator"""

    def create_agent(self, **kwargs) -> Agent:
        agent = Agent.create(environment=self,
                             **kwargs)

        print(f'Created {agent.__class__.__name__} agent.')
        return agent

    def learn(self, agent: Agent, num_episodes: int, max_episode_timesteps: int, weights_dir='weights/agents',
              agent_name='carla-agent', record_dir='data/recordings', skip_frames=10):
        record_path = None
        should_record = isinstance(record_dir, str)
        should_save = isinstance(weights_dir, str)

        # TODO: introduce callbacks: 'on_episode_start', 'on_episode_end', 'on_update', 'on_record', ..,
        for episode in range(num_episodes):
            states = self.reset()
            total_reward = 0.0

            if should_record:
                record_path = env_utils.get_record_path(base_dir=record_dir)
                print(f'Recording in {record_path}.')

            with self.synchronous_context:
                self.skip(num_frames=skip_frames)

                for i in range(max_episode_timesteps):
                    actions = agent.act(states)
                    states, terminal, reward = self.execute(actions, record_path=record_path)

                    total_reward += reward
                    terminal = terminal or (i == max_episode_timesteps - 1)

                    update_performed = agent.observe(reward, terminal)

                    if update_performed:
                        print(f'{i}/{max_episode_timesteps} -> update performed.')

                    if terminal:
                        print(f'Episode-{episode} ended, total_reward: {round(total_reward, 2)}\n')
                        break

            if should_save:
                env_utils.save_agent(agent, agent_name, directory=weights_dir)
                print('Agent saved.')

    def run(self, agent_args: dict, **kwargs):
        agent = self.create_agent(**agent_args)
        print('Agent created.')

        try:
            if kwargs.pop('load_agent', False):
                agent.load(directory=kwargs.get('weights_dir', 'weights/agents'),
                           filename=kwargs.get('agent_name', 'carla-agent'),
                           environment=self)
                print('Agent loaded.')

            self.learn(agent, **kwargs)

        finally:
            self.close()
            pygame.quit()


# -------------------------------------------------------------------------------------------------
# -- Baseline Experiments
# -------------------------------------------------------------------------------------------------

class CARLABaselineExperiment(CARLAExperiment):
    ACTIONS_SPEC = dict(type='float', shape=(3,), min_value=-1.0, max_value=1.0)
    DEFAULT_ACTIONS = np.array([0., 0., 0.])

    # vehicle: speed, accelerometer (x, y, z), gyroscope (x, y, z), position (x, y), destination (x, y), compass
    VEHICLE_FEATURES_SPEC = dict(type='float', shape=(12,))

    # road: intersection (bool), junction (bool), speed_limit, traffic_light (presence + state), lane_width,
    # lane_change, left_lane, right_lane
    ROAD_FEATURES_SPEC = dict(type='float', shape=(10,))

    DEFAULT_SENSORS = dict(imu=SensorSpecs.imu(),
                           collision=SensorSpecs.collision_detector(),
                           camera=SensorSpecs.rgb_camera(position='front',
                                                         attachment_type='Rigid',
                                                         image_size_x=670, image_size_y=500,
                                                         sensor_tick=1.0 / 30))

    def create_agent(self, max_episode_timesteps=512, batch_size=256, update_frequency=64, horizon=200, discount=0.997,
                     exploration=0.0, entropy=0.05) -> Agent:
        return super().create_agent(agent='tensorforce',
                                    max_episode_timesteps=max_episode_timesteps,

                                    policy=Specs.policy(distributions='gaussian',
                                                        network=Specs.agent_network_v0(),
                                                        temperature=0.99),
                                    optimizer=dict(type='adam', learning_rate=3e-4),

                                    objective=Specs.obj.policy_gradient(clipping_value=0.2),

                                    update=Specs.update(unit='timesteps', batch_size=batch_size,
                                                        frequency=update_frequency),

                                    reward_estimation=dict(horizon=horizon,
                                                           discount=discount,
                                                           estimate_advantage=True),
                                    exploration=exploration,
                                    entropy_regularization=entropy)

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

    def _action_penalty(self, actions, alpha=1, epsilon=0.01):
        return 0.0

    def _actions_to_control(self, actions):
        # Throttle
        # if actions[0] < 0:
        #     self.control.throttle = 0.0
        #     self.control.brake = 1.0
        # else:
        #     self.control.throttle = 1.0
        #     self.control.brake = 0.0

        if actions[0] < -0.33:
            self.control.throttle = 0.3
        elif actions[0] > 0.33:
            self.control.throttle = 0.9
        else:
            self.control.throttle = 0.5

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

class CARLAExperimentV0(CARLAExperiment):
    # skill, throttle/brake intensity, steer
    ACTIONS_SPEC = dict(type='float', shape=(3,), min_value=-1.0, max_value=1.0)
    DEFAULT_ACTIONS = np.array([0.0, 0.0, 0.0])

    DEFAULT_SENSORS = dict(imu=SensorSpecs.imu(),
                           collision=SensorSpecs.collision_detector(),
                           camera=SensorSpecs.rgb_camera(position='front',
                                                         attachment_type='Rigid',
                                                         image_size_x=670, image_size_y=500,
                                                         sensor_tick=1.0 / 30))

    def create_agent(self, batch_size=256, update_frequency=256, decay_steps=768, filters=36, decay=0.995, lr=0.1,
                     units=(256, 128), layers=(2, 2), temperature=(0.9, 0.7), exploration=0.0) -> Agent:
        ExpDecay = Specs.exp_decay
        policy_net = Specs.agent_network(conv=dict(stride=1, pooling='max', filters=filters),
                                         final=dict(layers=layers[0], units=units[0], activation='leaky-relu'))

        decay_lr = ExpDecay(steps=decay_steps, unit='updates', initial_value=lr, rate=decay)

        critic_net = Specs.agent_network(conv=dict(stride=1, pooling='max', filters=filters),
                                         final=dict(layers=layers[1], units=units[1]))

        return super().create_agent(policy=dict(network=policy_net,
                                                optimizer=dict(type='evolutionary', num_samples=6,
                                                               learning_rate=decay_lr),
                                                temperature=temperature[0]),

                                    batch_size=batch_size,
                                    update_frequency=update_frequency,

                                    critic=dict(network=critic_net,
                                                optimizer=dict(type='adam', learning_rate=3e-3),
                                                temperature=temperature[1]),

                                    discount=1.0,
                                    horizon=100,

                                    preprocessing=dict(image=[dict(type='image', width=140, height=105, grayscale=True),
                                                              dict(type='exponential_normalization')]),

                                    summarizer=Specifications.summarizer(frequency=update_frequency),

                                    entropy_regularization=ExpDecay(steps=decay_steps, unit='updates', initial_value=lr,
                                                                    rate=decay),
                                    exploration=exploration)

    def _action_penalty(self, actions, alpha=1, epsilon=0.01):
        return 0.0

    def _actions_to_control(self, actions):
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


class CARLAExperiment1(SynchronousCARLAEnvironment):
    # skill, throttle or brake, steer, reverse
    ACTIONS_SPEC = dict(type='float', shape=(4,), min_value=-1.0, max_value=1.0)
    DEFAULT_ACTIONS = np.array([0.0, 0.0, 0.0, 0.0])

    DEFAULT_SENSORS = dict(imu=SensorSpecs.imu(),
                           collision=SensorSpecs.collision_detector(),
                           camera=SensorSpecs.rgb_camera(position='front',
                                                         attachment_type='Rigid',
                                                         image_size_x=200, image_size_y=150))

    def _actions_to_control(self, actions):
        self.control.steer = float(actions[2])

        # throttle and brake are mutual exclusive:
        self.control.throttle = float(actions[1]) if actions[1] > 0 else 0.0
        self.control.brake = float(-actions[1]) if actions[1] < 0 else 0.0

        # reverse could be enabled only if throttle > 0
        if self.control.throttle > 0:
            self.control.reverse = bool(actions[3] > 0)
        else:
            self.control.reverse = False


class CARLAExperiment2(SynchronousCARLAEnvironment):
    # skill, throttle/brake intensity, steer
    ACTIONS_SPEC = dict(type='float', shape=(3,), min_value=-1.0, max_value=1.0)
    DEFAULT_ACTIONS = np.array([0.0, 0.0, 0.0])

    DEFAULT_SENSORS = dict(imu=SensorSpecs.imu(),
                           collision=SensorSpecs.collision_detector(),
                           camera=SensorSpecs.segmentation_camera(position='front',
                                                                  attachment_type='Rigid',
                                                                  image_size_x=200, image_size_y=150))

    # disable action penalty
    def _action_penalty(self, actions, alpha=1, epsilon=0.01):
        return 0.0

    def _actions_to_control(self, actions):
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


class CARLAExperiment3(CARLAExperiment2):
    DEFAULT_SENSORS = dict(imu=SensorSpecs.imu(),
                           collision=SensorSpecs.collision_detector(),
                           camera=SensorSpecs.rgb_camera(position='front',
                                                         attachment_type='Rigid',
                                                         # image_size_x=200, image_size_y=150,
                                                         image_size_x=670, image_size_y=500,
                                                         sensor_tick=1.0 / 30))


class CARLAExperiment4(CARLAExperiment3):

    def _get_observation(self, image):
        if image is None:
            image = np.zeros(shape=self.image_shape, dtype=np.uint8)

        if image.shape != self.image_shape:
            image = env_utils.resize(image, size=self.image_size)

        return dict(image=(2 * image - 255.0) / 255.0,
                    vehicle_features=self._get_vehicle_features(),
                    road_features=self._get_road_features(),
                    previous_actions=self.prev_actions)


# -------------------------------------------------------------------------------------------------
# -- Curriculum Learning Experiment
# -------------------------------------------------------------------------------------------------

class CurriculumCARLAEnvironment(CARLAExperiment3):

    def learn(self, agent: Agent, initial_timesteps: int, difficulty=1, increment=5, num_stages=1, max_timesteps=1024,
              trials_per_stage=5, max_repetitions=1, save_agent=True, load_agent=False, agent_name='carla-agent'):
        initial_difficulty = difficulty
        target_difficulty = initial_difficulty + num_stages * increment

        if load_agent:
            agent.load(directory='weights/agents', filename=agent_name, environment=self)
            print('Agent loaded.')

        for difficulty in range(initial_difficulty, target_difficulty + 1, increment):
            for r in range(max_repetitions):
                success_rate, avg_reward = self.stage(agent,
                                                      trials=trials_per_stage,
                                                      difficulty=difficulty,
                                                      max_timesteps=min(initial_timesteps * difficulty, max_timesteps))

                print(f'[D-{difficulty}] success_rate: {round(success_rate, 2)}, avg_reward: {round(avg_reward, 2)}')

                if save_agent:
                    agent.save(directory='weights/agents', filename=agent_name)
                    print(f'[D-{difficulty}] Agent saved.')

                print(f'Repetition {r}-D-{difficulty} ended.')

    def stage(self, agent: Agent, trials: int, difficulty: int, max_timesteps: int):
        assert trials > 0
        assert difficulty > 0
        assert max_timesteps > 0

        # self.reset(soft=False, route_size=difficulty)
        avg_reward = 0.0
        success_count = 0

        for trial in range(trials):
            # states = self.reset(soft=trial != 0, route_size=difficulty)
            states = self.reset(route_size=difficulty)
            trial_reward = 0.0

            with self.synchronous_context:
                for t in range(max_timesteps):
                    actions = agent.act(states)
                    states, terminal, reward = self.execute(actions, distance_threshold=3.0)

                    trial_reward += reward
                    terminal = terminal or (t == max_timesteps - 1)

                    if self.is_at_destination():
                        agent.observe(reward, terminal=True)
                        success_count += 1
                        print(f'[T-{trial}] Successful -> reward: {round(trial_reward, 2)}')
                        break

                    elif terminal:
                        agent.observe(reward, terminal=True)
                        print(f'[T-{trial}] not successful -> reward: {round(trial_reward, 2)}')
                        break
                    else:
                        agent.observe(reward, terminal=False)

            avg_reward += trial_reward

        return success_count / trials, avg_reward / trials

    def is_at_destination(self, distance_threshold=2.0):
        return self.route.distance_to_destination() < distance_threshold

    def _get_observation(self, image):
        if image is None:
            image = np.zeros(shape=self.image_shape, dtype=np.uint8)

        if image.shape != self.image_shape:
            image = env_utils.resize(image, size=self.image_size)

        return dict(image=image / 255.0,
                    vehicle_features=self._get_vehicle_features(),
                    road_features=self._get_road_features(),
                    previous_actions=self.prev_actions)
