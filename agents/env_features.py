"""Collection of features classes for building more complex CARLAEnvironments"""

import numpy as np
import carla

from typing import Tuple, List, Union, Optional, TypedDict


class ActionPenalty(object):
    """Feature: Skills and action_penalty."""
    # Skills: high-level actions
    SKILLS = {0: 'wait', 1: 'brake',
              2: 'steer right', 3: 'steer left',
              4: 'forward', 5: 'forward left', 6: 'forward right',
              7: 'backward', 8: 'backward left', 9: 'backward right'}

    DEFAULT_SKILL = np.array([0.0], dtype=np.float32)
    SKILL_SPEC = dict(type='float', shape=1, min_value=0.0, max_value=len(SKILLS) - 1.0)

    def get_skill_name(self, actions: dict):
        """Returns skill's name"""
        index = round(actions['skill'][0])
        return self.SKILLS[index]

    @staticmethod
    def action_penalty(actions: dict, eps=0.05) -> float:
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


class TemporalFeature(object):
    """Wraps a np.ndarray, allows to stack (concatenate) features along an additional "temporal" axis"""

    def __init__(self, horizon: int, shape: Union[tuple, int], default=0.0, dtype=np.float32, axis=None):
        assert isinstance(horizon, int)
        assert isinstance(shape, (tuple, int))
        assert isinstance(default, (int, float))

        self._time_horizon = horizon
        self._index = 0

        if isinstance(shape, int):
            self.shape = (horizon * shape,)
        elif axis == -1:
            self.shape = shape + (horizon,)
        else:
            self.shape = (horizon,) + shape

        self.dtype = dtype
        self.default = np.full(shape=self.shape, fill_value=default, dtype=dtype)
        self.data = self.default.copy()

    def append(self, value, depth=False):
        """Inserts a given new value at the given position in a circular fashion."""
        assert value is not None
        self._index = (self._index + 1) % self._time_horizon

        if depth:
            self.data[:, :, self._index] = value
        else:
            self.data[self._index] = value

    def reset(self):
        """Copies default to data (uses np.copyto)"""
        np.copyto(dst=self.data, src=self.default, casting='no')
        self._index = 0


class SkipTrick(object):

    @staticmethod
    def augment_states(states_spec: dict):
        # TODO: Include 'skill' in the state space, consider eventual termporal-horizon.
        pass
