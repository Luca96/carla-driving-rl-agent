"""A Human driver TensorForce agent mainly used for pretraining"""

import numpy as np
import time
import pygame

from tensorforce.agents import TensorforceAgent

from worlds import World
from worlds.controllers import BasicAgentController


def scale(n, range=(0.0, 1.0), to=(-1.0, +1.0)):
    return np.interp(n, range, to)


class HumanDriverAgent(TensorforceAgent):
    def __init__(self, environment, **kwargs):
        super().__init__(states=environment.states(), actions=environment.actions(), **kwargs)
        print('HumanDriverAgent')
        self.environment = environment
        self.controller = None

    def act(self, states, **kwargs):
        super().act(states, **kwargs)

        # get action by controller
        control = self.controller.act()
        actions = self._control_to_actions(control)

        # overwrite model's actions with manual controller's actions
        for name, _ in self.actions_spec.items():
            self.actions_buffers[name][0, 0] = actions

        return actions

    def init_controller(self):
        print('human.init_controller')
        self.controller = BasicAgentController(vehicle=self.environment.world.player,
                                               destination=self.environment.world.target_position.location,
                                               target_speed=30)

    @staticmethod
    def _control_to_actions(control):
        print(control)
        return [scale(control.throttle),
                float(control.steer),
                scale(control.brake),
                scale(control.reverse) * -1.0]
