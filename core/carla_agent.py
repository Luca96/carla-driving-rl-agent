import os
import gym
import tensorflow as tf

from typing import Union

from rl import utils
from rl.agents import PPOAgent
from rl.environments import ThreeCameraCARLAEnvironmentDiscrete

from core.networks import CARLANetwork


class FakeCARLAEnvironment(gym.Env):
    """A testing-only environment with the same state- and action-space of a CARLA Environment"""

    def __init__(self):
        super().__init__()
        env = ThreeCameraCARLAEnvironmentDiscrete

        self.action_space = env.ACTION['space']
        self.observation_space = gym.spaces.Dict(road=env.ROAD_FEATURES['space'],
                                                 vehicle=env.VEHICLE_FEATURES['space'],
                                                 past_control=env.CONTROL['space'], command=env.COMMAND_SPACE,
                                                 image=gym.spaces.Box(low=-1.0, high=1.0, shape=(90, 360, 3)))

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass


class CARLAgent(PPOAgent):
    # predefined architecture:
    DEFAULT_CONTROL = dict(num_layers=4, units_multiplier=16)
    DEFAULT_DYNAMICS = dict(road=dict(num_layers=3, units=16),
                            vehicle=dict(num_layers=2, units=16),
                            command=dict(num_layers=2, units=16),
                            shufflenet=dict(linear_units=128, g=0.5, last_channels=768),
                            value=dict(linear_units=0, units=8),
                            action=dict(linear_units=32, units=16))

    def __init__(self, *args, context_size=64, name='carla-ppo', **kwargs):
        network_spec = kwargs.get('network', {})
        network_spec.setdefault('network', CARLANetwork)
        network_spec.setdefault('context_size', context_size)
        network_spec.setdefault('control', self.DEFAULT_CONTROL)
        network_spec.setdefault('dynamics', self.DEFAULT_DYNAMICS)

        super().__init__(*args, name=name, network=network_spec, **kwargs)

        self.weights_path['dynamics'] = os.path.join(self.base_path, 'dynamics_model')

    def act(self, state: dict):
        raise NotImplementedError


if __name__ == '__main__':
    agent = CARLAgent(FakeCARLAEnvironment(), batch_size=32, log_mode=None)
    agent.summary()
    pass
