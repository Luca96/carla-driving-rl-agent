
import carla
import pygame

from typing import Optional
from pygame.constants import K_q, K_UP, K_w, K_LEFT, K_a, K_RIGHT, K_d, K_DOWN, K_s, K_SPACE, K_ESCAPE, KMOD_CTRL

from tensorforce import Agent

from agents.specifications import Specifications as Specs
from agents.environment import SynchronousCARLAEnvironment

from navigation import LocalPlanner


class Dummy:
    @staticmethod
    def random_walk(carla_env: SynchronousCARLAEnvironment, max_episode_timesteps: int, speed: float, **kwargs):
        return RandomWalkAgent(carla_env, max_episode_timesteps, speed, **kwargs)

    @staticmethod
    def keyboard(carla_env: SynchronousCARLAEnvironment, fps=30.0, mode='play'):
        return KeyboardAgent(carla_env, fps, mode)


class Agents:
    """Provides predefined agents"""
    dummy = Dummy

    @staticmethod
    def get(kind: str, env: SynchronousCARLAEnvironment, *args, **kwargs):
        pass

    @staticmethod
    def baseline(carla_env: SynchronousCARLAEnvironment, max_episode_timesteps=512, batch_size=256, exploration=0.0,
                 update_frequency: Optional[int] = None, horizon: Optional[int] = None, discount=0.997, entropy=0.05,
                 **kwargs) -> Agent:
        horizon = horizon or (batch_size - 1)
        assert horizon < batch_size
        update_frequency = update_frequency or batch_size

        return Agent.create(agent='tensorforce',
                            environment=carla_env,
                            max_episode_timesteps=max_episode_timesteps,

                            policy=Specs.policy(distributions='gaussian',
                                                network=Specs.agent_network_v0(),
                                                temperature=0.99),

                            optimizer=dict(type='adam', learning_rate=3e-4),
                            objective=Specs.obj.policy_gradient(clipping_value=0.2),
                            update=Specs.update(unit='timesteps', batch_size=batch_size, frequency=update_frequency),

                            reward_estimation=dict(horizon=horizon,
                                                   discount=discount,
                                                   estimate_advantage=True),
                            exploration=exploration,
                            entropy_regularization=entropy,
                            **kwargs)

    @staticmethod
    def evolutionary(carla_env: SynchronousCARLAEnvironment, max_episode_timesteps: int, batch_size=256, num_samples=6,
                     update_frequency: Optional[int] = None, decay_steps=768, filters=36, decay=0.995, lr=0.1,
                     units=(256, 128), layers=(2, 2), temperature=(0.9, 0.7), horizon: Optional[int] = None, width=140,
                     height=105,  **kwargs) -> Agent:
        horizon = horizon or (batch_size - 1)
        assert horizon < batch_size

        policy_net = Specs.agent_network(conv=dict(stride=1, pooling='max', filters=filters),
                                         final=dict(layers=layers[0], units=units[0], activation='leaky-relu'))

        decay_lr = Specs.exp_decay(steps=decay_steps, unit='updates', initial_value=lr, rate=decay)

        critic_net = Specs.agent_network(conv=dict(stride=1, pooling='max', filters=filters),
                                         final=dict(layers=layers[1], units=units[1]))

        if 'preprocessing' in kwargs.keys():
            preprocessing = kwargs.pop('preprocessing')
        else:
            preprocessing = dict(image=[dict(type='image', width=width, height=height, grayscale=True),
                                        dict(type='exponential_normalization')])

        return Specs.carla_agent(carla_env, max_episode_timesteps,
                                 policy=dict(network=policy_net,
                                             optimizer=dict(type='evolutionary', num_samples=num_samples,
                                                            learning_rate=decay_lr),
                                             temperature=temperature[0]),

                                 batch_size=batch_size,
                                 update_frequency=update_frequency or batch_size,

                                 critic=dict(network=critic_net,
                                             optimizer=dict(type='adam', learning_rate=3e-3),
                                             temperature=temperature[1]),
                                 discount=1.0,
                                 horizon=horizon,

                                 preprocessing=preprocessing,

                                 summarizer=Specs.summarizer(frequency=update_frequency),

                                 entropy_regularization=Specs.exp_decay(steps=decay_steps, unit='updates',
                                                                        initial_value=lr, rate=decay),
                                 **kwargs)

    @staticmethod
    def criticless(carla_env: SynchronousCARLAEnvironment, max_episode_timesteps: int, batch_size=256, num_samples=6,
                   update_frequency: Optional[int] = None, decay_steps=768, filters=36, decay=0.995, lr=0.1, units=256,
                   layers=2, temperature=0.9, horizon: Optional[int] = None, width=140, height=105, discount=1.0,
                   **kwargs) -> Agent:
        horizon = horizon or (batch_size - 1)
        assert horizon < batch_size
        update_frequency = update_frequency or batch_size

        policy_net = Specs.agent_network(conv=dict(stride=1, pooling='max', filters=filters),
                                         final=dict(layers=layers, units=units, activation='leaky-relu'))

        decay_lr = Specs.exp_decay(steps=decay_steps, unit='updates', initial_value=lr, rate=decay)

        if 'preprocessing' in kwargs.keys():
            preprocessing = kwargs.pop('preprocessing')
            print(preprocessing)
        else:
            preprocessing = dict(image=[dict(type='image', width=width, height=height, grayscale=True),
                                        dict(type='exponential_normalization')])

        return Agent.create(agent='tensorforce',
                            environment=carla_env,
                            max_episode_timesteps=max_episode_timesteps,

                            policy=Specs.policy(distributions='gaussian',
                                                network=policy_net,
                                                temperature=temperature),

                            optimizer=dict(type='evolutionary', num_samples=num_samples, learning_rate=decay_lr),
                            objective=Specs.obj.policy_gradient(clipping_value=0.2),
                            update=Specs.update(unit='timesteps', batch_size=batch_size, frequency=update_frequency),

                            reward_estimation=dict(horizon=horizon,
                                                   discount=discount,
                                                   estimate_advantage=True),

                            preprocessing=preprocessing,

                            entropy_regularization=Specs.exp_decay(steps=decay_steps, unit='updates', initial_value=lr,
                                                                   rate=decay),
                            **kwargs)

    @staticmethod
    def ppo_like(carla_env: SynchronousCARLAEnvironment, max_episode_timesteps: int, policy: dict,
                 critic: Optional[dict] = None, batch_size=64, update_frequency: Optional[int] = None, huber_loss=0.0,
                 discount=1.0, horizon: Optional[int] = None, estimate_terminal=False, **kwargs) -> Agent:
        horizon = horizon or (batch_size - 1)
        assert horizon < batch_size
        update_frequency = update_frequency or batch_size

        if critic is None:
            critic_policy = critic_optimizer = critic_objective = None
        else:
            critic_policy = dict(network=critic.get('network', None))
            critic_optimizer = critic.get('optimizer', dict(type='adam', learning_rate=3e-3))
            critic_objective = Specs.obj.value(value='state', huber_loss=huber_loss)

        return Agent.create(agent='tensorforce',
                            environment=carla_env,
                            max_episode_timesteps=max_episode_timesteps,
                            update=Specs.update(unit='timesteps', batch_size=batch_size, frequency=update_frequency),

                            # Policy
                            policy=Specs.policy(network=policy.get('network'),
                                                distributions=policy.get('distributions', None),
                                                temperature=policy.get('temperature', 1.0),
                                                infer_states_value=policy.get('infer_states', False)),
                            memory=dict(type='recent'),
                            optimizer=policy.get('optimizer', dict(type='adam', learning_rate=3e-4)),
                            objective=Specs.obj.policy_gradient(clipping_value=0.2, ratio_based=True),

                            # Critic
                            baseline_policy=critic_policy,
                            baseline_optimizer=critic_optimizer,
                            baseline_objective=critic_objective,

                            # Reward
                            reward_estimation=dict(discount=discount,
                                                   horizon=horizon,
                                                   estimate_horizon=False if critic is None else 'early',
                                                   estimate_terminal=estimate_terminal,
                                                   estimate_advantage=True),
                            **kwargs)


# -------------------------------------------------------------------------------------------------
# -- Dummy Agents
# -------------------------------------------------------------------------------------------------

class DummyAgent(object):

    def reset(self):
        pass

    def act(self, states, **kwargs):
        raise NotImplementedError

    def observe(self, reward, terminal=False, **kwargs):
        raise NotImplementedError

    def load(self, **kwargs):
        raise NotImplementedError('Unsupported operation!')

    def save(self, **kwargs):
        raise NotImplementedError('Unsupported operation!')


class RandomWalkAgent(DummyAgent):
    """A dummy agent whose only purpose is to record traces for pretraining other agents"""

    def __init__(self, env: SynchronousCARLAEnvironment, max_episode_timesteps: int, speed=30.0, use_speed_limit=False,
                 traces_dir=None, **kwargs):
        self.agent = Agent.create(agent='constant',
                                  environment=env,
                                  max_episode_timesteps=max_episode_timesteps,
                                  action_values=env.DEFAULT_ACTIONS,
                                  recorder=dict(directory=traces_dir) if isinstance(traces_dir, str) else None,
                                  **kwargs)
        self.env = env
        self.index = 0
        self.use_speed_limit = use_speed_limit

        # Behaviour planner:
        self.options = dict(target_speed=speed,
                            lateral_control_dict={'K_P': 1, 'K_D': 0.02, 'K_I': 0, 'dt': 1.0 / self.env.fps})

        self.local_planner = None

    def reset(self):
        self.index = 0
        self.local_planner.reset_vehicle()
        self.local_planner = LocalPlanner(vehicle=self.env.vehicle, opt_dict=self.options)

    def act(self, states, **kwargs):
        if self.local_planner is None:
            self.local_planner = LocalPlanner(vehicle=self.env.vehicle, opt_dict=self.options)

        if self.use_speed_limit:
            self.local_planner.set_speed(speed=self.env.vehicle.get_speed_limit())

        # hack: records states
        _ = self.agent.act(states, **kwargs)

        control = self._run_step()
        actions, skill_name = self.env.control_to_actions(control)

        # hack: records "custom" (not constant) actions
        for name in self.agent.actions_spec.keys():
            if isinstance(actions, dict):
                self.agent.actions_buffers[name][0, self.index] = actions[name]
            else:
                self.agent.actions_buffers[name][0, self.index] = actions

        self.index += 1
        return actions

    def load(self, **kwargs):
        super().load(**kwargs)

    def save(self, **kwargs):
        super().save(**kwargs)

    def observe(self, reward, terminal=False, **kwargs):
        # hack: records rewards and terminals
        return self.agent.observe(reward, terminal=terminal, **kwargs)

    def _run_step(self) -> carla.VehicleControl:
        """Execute one step of navigation. WARNING: does not check for obstacles and traffic lights!
            :return: carla.VehicleControl
        """
        return self.local_planner.run_step(debug=False)


class KeyboardAgent(DummyAgent):
    """Two modes: 'play', and 'pretrain/record'
        - in "play mode" a human controls the given agent with a keyboard,
        - in "pretrain mode" a human controls the agent with a keyboard and records its actions plus the states.
    """

    # TODO: implement 'record' mode
    def __init__(self, env: SynchronousCARLAEnvironment, fps=30.0, mode='play'):
        self.env = env
        self.mode = mode
        self.fps = fps

        self.control = carla.VehicleControl()
        self._steer_cache = 0.0

    def reset(self):
        self.control = carla.VehicleControl()
        self._steer_cache = 0.0

    def act(self, states, **kwargs):
        return self._parse_events()

    def observe(self, reward, terminal=False, **kwargs):
        return False

    def _parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    self.close()

                elif event.key == K_q:
                    self.control.gear = 1 if self.control.reverse else -1

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
        self.control.reverse = self.control.gear < 0

        # actions
        throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer = round(self._steer_cache, 1)
        brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        reverse = 1.0 if self.control.reverse else 0.0
        hand_brake = keys[K_SPACE]

        return [throttle, steer, brake, reverse, hand_brake]

    def close(self):
        raise Exception('closing...')

    def load(self, **kwargs):
        super().load(**kwargs)

    def save(self, **kwargs):
        super().save(**kwargs)
