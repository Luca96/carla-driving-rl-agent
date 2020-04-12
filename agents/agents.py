
import carla
import numpy as np

from tensorforce import Agent
from tensorforce.agents import ConstantAgent, RandomAgent

from agents.specifications import Specifications as Specs
from agents.learn.environment import SynchronousCARLAEnvironment

from worlds.navigation.local_planner import LocalPlanner


class Dummy:
    @staticmethod
    def random_walk(carla_env: SynchronousCARLAEnvironment, max_episode_timesteps: int, speed: float, **kwargs):
        return RandomWalkAgent(carla_env, max_episode_timesteps, speed, **kwargs)


class Agents:
    """Provides predefined agents"""
    dummy = Dummy

    @staticmethod
    def baseline(carla_env: SynchronousCARLAEnvironment, max_episode_timesteps=512, batch_size=256, update_frequency=64,
                 horizon=200, discount=0.997, exploration=0.0, entropy=0.05, **kwargs) -> Agent:
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
                     update_frequency=256, decay_steps=768, filters=36, decay=0.995, lr=0.1, units=(256, 128),
                     layers=(2, 2), temperature=(0.9, 0.7), horizon=100, width=140, height=105,  **kwargs) -> Agent:

        policy_net = Specs.agent_network(conv=dict(stride=1, pooling='max', filters=filters),
                                         final=dict(layers=layers[0], units=units[0], activation='leaky-relu'))

        decay_lr = Specs.exp_decay(steps=decay_steps, unit='updates', initial_value=lr, rate=decay)

        critic_net = Specs.agent_network(conv=dict(stride=1, pooling='max', filters=filters),
                                         final=dict(layers=layers[1], units=units[1]))

        return Specs.carla_agent(carla_env, max_episode_timesteps,
                                 policy=dict(network=policy_net,
                                             optimizer=dict(type='evolutionary', num_samples=num_samples,
                                                            learning_rate=decay_lr),
                                             temperature=temperature[0]),

                                 batch_size=batch_size,
                                 update_frequency=update_frequency,

                                 critic=dict(network=critic_net,
                                             optimizer=dict(type='adam', learning_rate=3e-3),
                                             temperature=temperature[1]),
                                 discount=1.0,
                                 horizon=horizon,

                                 preprocessing=dict(image=[dict(type='image', width=width, height=height, grayscale=True),
                                                           dict(type='exponential_normalization')]),

                                 summarizer=Specs.summarizer(frequency=update_frequency),

                                 entropy_regularization=Specs.exp_decay(steps=decay_steps, unit='updates',
                                                                        initial_value=lr, rate=decay),
                                 **kwargs)


class RandomWalkAgent(object):
    """A dummy agent whose only purpose is to record traces of the entire map for pretraining other agents"""

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

    def observe(self, reward, terminal=False, **kwargs):
        # hack: records rewards and terminals
        return self.agent.observe(reward, terminal=terminal, **kwargs)

    def load(self, **kwargs):
        raise NotImplementedError('Unsupported operation!')

    def save(self, **kwargs):
        raise NotImplementedError('Unsupported operation!')

    def _run_step(self) -> carla.VehicleControl:
        """Execute one step of navigation. WARNING: does not check for obstacles and traffic lights!
            :return: carla.VehicleControl
        """
        return self.local_planner.run_step(debug=False)


class MapExplorationAgent(Agent):
    """A dummy agent whose only purpose is to record traces for pretraining other agents"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass
