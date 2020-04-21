"""Planners"""

import carla


# TODO: WIP
class RoutePlanner(object):
    """Abstract route Planner"""
    def __init__(self, carla_map: carla.Map):
        self.map = carla_map
        self.route = []

    def plan(self, *args, **kwargs):
        raise NotImplementedError


class RandomPlanner(RoutePlanner):
    """A planner that selects random waypoints as the vehicle goes further"""
    def __init__(self, carla_map: carla.Map, resolution=2.0, window=5):
        super().__init__(carla_map)
        self.resolution = resolution
        self.window = window

    def plan(self, *args, **kwargs):
        pass
