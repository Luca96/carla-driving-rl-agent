import math
import random
import carla

import worlds.debug

from worlds.navigation import RoutePlanner
from worlds.utils import l2_norm, Object, Colors


class Route(object):
    """Wraps a list of [(carla.Waypoint, RoadOption)] returned by a RoutePlanner."""
    def __init__(self, planner: RoutePlanner):
        self.planner = planner
        self.planner.setup()

        self.path = None
        self.size = 0.0
        self.closest_path = None

    def __len__(self):
        if self.path is None:
            return 0

        return len(self.path)

    def plan(self, origin: carla.Location, destination: carla.Location):
        """Calculates the sequence of waypoints (a route) from origin to destination."""
        self.path = self.planner.trace_route(origin, destination)
        self._compute_route_size()

        self.closest_path = Object(index=0, distance=0.0, waypoint=self.path[0][0])

    def random_plan(self, origin: carla.Location, length: int, distance=2.0):
        self.path = self.planner.trace_random_route(origin, length, distance)
        self._compute_route_size()

        self.closest_path = Object(index=0, distance=0.0, waypoint=self.path[0][0])

    def update_closest_waypoint(self, location: carla.Location):
        """Returns the closest route's Waypoint to the given location (carla.Location) that advances the completion
        of the planned route. """
        closest_waypoint = self.closest_path
        closest_waypoint.distance = math.inf

        # TODO: takes linear time in len(route), can speedup with binary search?
        for i in range(closest_waypoint.index, len(self.path)):
            waypoint = self.path[i][0]
            distance_to_location = l2_norm(location1=location,
                                           location2=waypoint.transform.location)

            if distance_to_location < closest_waypoint.distance:
                closest_waypoint.index = i
                closest_waypoint.distance = distance_to_location
                closest_waypoint.waypoint = waypoint

        assert closest_waypoint.index >= 0
        self.closest_path = closest_waypoint

    def draw_closest_waypoint(self, world_debug: carla.DebugHelper, location: carla.Location, life_time=1.0):
        """Draws the closest route's waypoint to the given location"""
        index = self.closest_path.index
        curr_waypoint = self.closest_path.waypoint.transform.location
        next_waypoint = self.path[index + 1][0].transform.location if index + 1 < len(self.path) else curr_waypoint

        world_debug.draw_arrow(location, curr_waypoint, arrow_size=0.3, life_time=life_time)
        world_debug.draw_arrow(location, next_waypoint, arrow_size=0.3, life_time=life_time)

    def draw_route(self, debug: carla.DebugHelper, life_time=1 / 30.0):
        for waypoint, road_op in self.path:
            worlds.debug.draw_transform(debug, waypoint.transform, Colors.green, life_time)
            debug.draw_string(waypoint.transform.location, road_op.name, False, Colors.orange, life_time)

    def distance_to_destination(self, location: carla.Location = None):
        """Returns the distance from the closest route's waypoint to the route's destination (last waypoint).
           If location is provided, it will call 'update_closest_waypoint',
        """
        if location is not None:
            self.update_closest_waypoint(location)

        distance = 0.0
        for i in range(self.closest_path.index + 1, len(self.path)):
            waypoint1 = self.path[i - 1][0].transform
            waypoint2 = self.path[i][0].transform

            distance += l2_norm(location1=waypoint1.location,
                                location2=waypoint2.location)

        return distance

    def get_closest_waypoint_location(self):
        return self.closest_path.waypoint.transform.location

    def _compute_route_size(self):
        """Compute the size (in meters) of the entire planned route"""
        self.size = 0.0

        for i in range(1, len(self.path)):
            self.size += l2_norm(location1=self.path[i - 1][0].transform.location,
                                 location2=self.path[i][0].transform.location)
