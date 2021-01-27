
import numpy as np

from enum import Enum


class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """

    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANE_FOLLOW = 4
    CHANGE_LANE_LEFT = 5
    CHANGE_LANE_RIGHT = 6

    @property
    def shape(self) -> tuple:
        return 6,

    def to_one_hot(self):
        """Returns a one-hot encoded route-option as np.array"""
        encoded = np.zeros(shape=self.shape, dtype=np.float32)

        if self.value == -1:
            return encoded

        # Put a 1 in the position specified by value (-1 is for 0-indexing)
        encoded[self.value - 1] = 1.0
        return encoded
