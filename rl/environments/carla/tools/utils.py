"""A collection of utility functions."""

import re
import time
import math
import carla
import numpy as np

from functools import wraps

# Globals:

find_weather_regex = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
cc = carla.ColorConverter
Attachment = carla.AttachmentType
epsilon = np.finfo(np.float32).eps


def find_weather_presets():
    name = lambda x: ' '.join(m.group(0) for m in find_weather_regex.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def l2_norm(location1: carla.Location, location2: carla.Location) -> float:
    """Computes the Euclidean distance between two carla.Location objects."""
    dx = location1.x - location2.x
    dy = location1.y - location2.y
    dz = location1.z - location2.z
    return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2) + epsilon


def unit_vector(location1: carla.Location, location2: carla.Location) -> list:
    """Returns the unit vector from location1 to location2."""
    x = location2.x - location1.x
    y = location2.y - location1.y
    z = location2.z - location1.z
    norm = np.linalg.norm([x, y, z]) + epsilon

    return [x / norm, y / norm, z / norm]


def vector_norm(vec: carla.Vector3D) -> float:
    """Returns the norm/magnitude (a scalar) of the given 3D vector."""
    return math.sqrt(vec.x**2 + vec.y**2 + vec.z**2)


def speed(actor: carla.Actor) -> float:
    """Returns the speed of the given actor in km/h."""
    return 3.6 * vector_norm(actor.get_velocity())


def dot_product(a: carla.Vector3D, b: carla.Vector3D) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z


def cosine_similarity(a: carla.Vector3D, b: carla.Vector3D) -> float:
    """-1: opposite vectors (pointing in the opposite direction),
        0: orthogonal,
        1: exactly the same (pointing in the same direction)
    """
    return dot_product(a, b) / (vector_norm(a) * vector_norm(b))


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


def profile(fn):
    # source: https://stackoverflow.com/questions/3620943/measuring-elapsed-time-with-the-time-module/46544199

    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.time()
        ret = fn(*args, **kwargs)

        elapsed_time = time.time() - start_time
        print(f'[PROFILE] <{fn.__name__}> takes {round(elapsed_time, 4)}ms.')

        return ret

    return with_profiling


# -------------------------------------------------------------------------------------------------
# -- Drawing stuff
# -------------------------------------------------------------------------------------------------

def draw_vehicle_trail(world, vehicle, tick_time=0.2, life_time=60):
    """Draws the path of an actor through the worlds, printing information at each waypoint.
        @:arg vehicle: see recipe 'actor_spectator'.
        @:arg tick_time: time to sleep between updating the drawing.
    """
    map = world.get_map()
    debug = world.debug

    lane_type = carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk
    current_w = map.get_waypoint(vehicle.get_location())

    while True:
        next_w = map.get_waypoint(vehicle.get_location(), lane_type=lane_type)

        # Check if the vehicle is moving
        if next_w.id != current_w.id:
            # Check if the vehicle is on a sidewalk
            if current_w.lane_type == carla.LaneType.Sidewalk:
                color = Colors.cyan if current_w.is_junction else Colors.red
                draw_waypoint_union(debug, current_w, next_w, color, life_time)
            else:
                color = Colors.cyan if current_w.is_junction else Colors.green
                draw_waypoint_union(debug, current_w, next_w, color, life_time)

            speed = 3.6 * vector_norm(vec=vehicle.get_velocity())
            debug.draw_string(current_w.transform.location, str('%15.0f km/h' % speed), False, Colors.orange, life_time)
            draw_transform(debug, current_w.transform, Colors.white, life_time)

        # Update the current waypoint and sleep for some time
        current_w = next_w
        time.sleep(tick_time)


def draw_route(debug, route, life_time=60):
    for waypoint, road_op in route:
        draw_transform(debug, waypoint.transform, Colors.green, life_time)
        debug.draw_string(waypoint.transform.location, road_op.name, False, Colors.orange, life_time)


def draw_bounding_box(world):
    """Shows how to draw traffic light actor bounding boxes from a worlds snapshot."""
    debug = world.debug
    world_snaphost = world.get_snapshot()

    for actor_snapshot in world_snaphost:
        actor = world.get_actor(actor_snapshot.id)

        if actor.type_id == 'traffic.traffic_light':
            actor_transform = actor_snapshot.get_transform()
            bounding_box = carla.BoundingBox(actor_transform.location, carla.Vector3D(0.5, 0.5, 2))
            debug.draw_box(bounding_box, actor_transform.rotation, 0.05, Colors.red, 0)


# https://github.com/carla-simulator/carla/blob/master/PythonAPI/util/lane_explorer.py
def draw_waypoint_union(debug, w0, w1, color=Colors.red, lt=5):
    debug.draw_line(w0.transform.location + carla.Location(z=0.25),
                    w1.transform.location + carla.Location(z=0.25),
                    thickness=0.1, color=color, life_time=lt, persistent_lines=False)

    debug.draw_point(w1.transform.location + carla.Location(z=0.25), 0.1, color, lt, False)


def draw_transform(debug, trans, col=Colors.red, lt=-1):
    yaw_in_rad = math.radians(trans.rotation.yaw)
    pitch_in_rad = math.radians(trans.rotation.pitch)

    p1 = carla.Location(x=trans.location.x + math.cos(pitch_in_rad) * math.cos(yaw_in_rad),
                        y=trans.location.y + math.cos(pitch_in_rad) * math.sin(yaw_in_rad),
                        z=trans.location.z + math.sin(pitch_in_rad))

    debug.draw_arrow(trans.location, p1, thickness=0.05, arrow_size=0.1, color=col, life_time=lt)
