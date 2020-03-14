"""A collection of utility functions."""

import re
import numpy as np
import carla

from worlds.sensors import *

# Globals:

find_weather_regex = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
cc = carla.ColorConverter
Attachment = carla.AttachmentType


def find_weather_presets():
    name = lambda x: ' '.join(m.group(0) for m in find_weather_regex.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def l2_norm(location1, location2):
    """Computes the Euclidean distance between two carla.Location objects."""
    return math.sqrt((location1.x - location2.x) ** 2 + (location1.y - location2.y) ** 2)


def unit_vector(location_1, location_2):
    """
    Returns the unit vector from location_1 to location_2.
        @:arg: location_1, location_2:   carla.Location objects
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps

    return [x / norm, y / norm, z / norm]


def vector_norm(vec: carla.Vector3D):
    """Returns the norm/magnitude (a scalar) of the given 3D vector."""
    return math.sqrt(vec.x**2 + vec.y**2 + vec.z**2)


class Colors(object):
    """Wraps some carla.Color instances."""
    red = carla.Color(255, 0, 0)
    green = carla.Color(0, 255, 0)
    blue = carla.Color(47, 210, 231)
    cyan = carla.Color(0, 255, 255)
    yellow = carla.Color(255, 255, 0)
    orange = carla.Color(255, 162, 0)
    white = carla.Color(255, 255, 255)
    black = carla.Color(0, 0, 0)


class Object(object):
    """Creates generic objects with fields given as named arguments."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# Use this dict to convert lanes objects to integers:
WAYPOINT_DICT = dict(lane_change={carla.LaneChange.NONE: 0,
                                  carla.LaneChange.Both: 1,
                                  carla.LaneChange.Left: 2,
                                  carla.LaneChange.Right: 3},
                     lane_type={carla.LaneType.NONE: 0,
                                carla.LaneType.Bidirectional: 1,
                                carla.LaneType.Biking: 2,
                                carla.LaneType.Border: 3,
                                carla.LaneType.Driving: 4,
                                carla.LaneType.Entry: 5,
                                carla.LaneType.Exit: 6,
                                carla.LaneType.Median: 7,
                                carla.LaneType.OffRamp: 8,
                                carla.LaneType.OnRamp: 9,
                                carla.LaneType.Parking: 10,
                                carla.LaneType.Rail: 11,
                                carla.LaneType.Restricted: 12,
                                carla.LaneType.RoadWorks: 13,
                                carla.LaneType.Shoulder: 14,
                                carla.LaneType.Sidewalk: 15,
                                carla.LaneType.Special1: 16,
                                carla.LaneType.Special2: 17,
                                carla.LaneType.Special3: 18,
                                carla.LaneType.Stop: 19,
                                carla.LaneType.Tram: 20,
                                carla.LaneType.Any: 21},
                     lane_marking_type={carla.LaneMarkingType.NONE: 0,
                                        carla.LaneMarkingType.BottsDots: 1,
                                        carla.LaneMarkingType.Broken: 2,
                                        carla.LaneMarkingType.BrokenBroken: 3,
                                        carla.LaneMarkingType.BrokenSolid: 4,
                                        carla.LaneMarkingType.Curb: 5,
                                        carla.LaneMarkingType.Grass: 6,
                                        carla.LaneMarkingType.Solid: 7,
                                        carla.LaneMarkingType.SolidBroken: 8,
                                        carla.LaneMarkingType.SolidSolid: 9,
                                        carla.LaneMarkingType.Other: 10},
                     traffic_light={carla.TrafficLightState.Green: 0,
                                    carla.TrafficLightState.Red: 1,
                                    carla.TrafficLightState.Yellow: 2,
                                    carla.TrafficLightState.Off: 3,
                                    carla.TrafficLightState.Unknown: 4}
                     )
