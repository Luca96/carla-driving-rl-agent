import math
import carla
import random

from rl.environments.carla.navigation import RoutePlanner
from rl.environments.carla.tools import utils
from rl.environments.carla.tools.utils import Object, Colors


class Route(object):
    """Wraps a list of [(carla.Waypoint, RoadOption)] returned by a RoutePlanner."""
    def __init__(self, planner: RoutePlanner):
        self.planner = planner
        self.planner.setup()

        self.path = None
        self.size = 0.0
        self.next = None

    def __len__(self):
        if self.path is None:
            return 0

        return len(self.path)

    def plan(self, origin: carla.Location, destination: carla.Location):
        """Calculates the sequence of waypoints (a route) from origin to destination."""
        self.path = self.planner.trace_route(origin, destination)
        self._compute_route_size()

        self.next = Object(index=0, distance=0.0, waypoint=self.path[0][0], road_op=self.path[0][1])

    def update_next_waypoint(self, location: carla.Location):
        """Returns the closest route's Waypoint to the given location (carla.Location) that advances the completion
           of the planned route.
        """
        closest = self.next
        closest.distance = math.inf

        # TODO: can speedup with binary search?
        for i in range(closest.index, len(self.path)):
            waypoint = self.path[i][0]
            distance_to_location = utils.l2_norm(location1=location,
                                                 location2=waypoint.transform.location)

            if distance_to_location < closest.distance:
                closest.index = i
                closest.distance = distance_to_location
                closest.waypoint = waypoint
                closest.road_op = self.path[i][1]

        assert closest.index >= 0
        self.next = closest

    def draw_next_waypoint(self, world_debug: carla.DebugHelper, location: carla.Location, life_time=1.0):
        """Draws the closest route's waypoint to the given location"""
        index = self.next.index
        curr_waypoint = self.next.waypoint.transform.location
        next_waypoint = self.path[index + 1][0].transform.location if index + 1 < len(self.path) else curr_waypoint

        world_debug.draw_arrow(location, curr_waypoint, arrow_size=0.3, life_time=life_time)
        world_debug.draw_arrow(location, next_waypoint, arrow_size=0.3, life_time=life_time)

    def draw_route(self, debug: carla.DebugHelper, life_time=1 / 30.0):
        for waypoint, road_op in self.path:
            debug.draw_string(waypoint.transform.location, road_op.name, False, Colors.orange, life_time)
            utils.draw_transform(debug, waypoint.transform, Colors.green, life_time)

    def distance_to_destination(self, location: carla.Location):
        """Returns the distance from the given location the route's destination (last waypoint)"""
        distance = utils.l2_norm(location1=location,
                                 location2=self.next.waypoint.transform.location)

        for i in range(self.next.index + 1, len(self.path)):
            waypoint1 = self.path[i - 1][0].transform
            waypoint2 = self.path[i][0].transform

            distance += utils.l2_norm(location1=waypoint1.location,
                                      location2=waypoint2.location)
        return distance

    def distance_to_next_waypoint(self):
        return self.next.distance

    def get_next_waypoint_location(self):
        return self.next.waypoint.transform.location

    def get_next_waypoints(self, amount: int) -> list:
        start = self.next.index
        end = min(start + amount, len(self.path))

        next_waypoints = map(lambda x: x[0], self.path[start:end])
        return list(next_waypoints)

    def random_waypoint(self):
        return random.choice(self.path)[0]

    def _compute_route_size(self):
        """Compute the size (in meters) of the entire planned route"""
        self.size = 0.0

        for i in range(1, len(self.path)):
            self.size += utils.l2_norm(location1=self.path[i - 1][0].transform.location,
                                       location2=self.path[i][0].transform.location)
