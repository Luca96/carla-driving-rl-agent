import math
import time

import carla

from functools import wraps
from worlds.utils import Colors, vector_norm


def profile(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.time()
        ret = fn(*args, **kwargs)

        elapsed_time = time.time() - start_time
        # print(f'[PROFILE] Function <{fn.__name__}> takes {round(elapsed_time / 1000.0, 4)}ms.')
        # print(f'[PROFILE] <{fn.__name__}> takes {round(elapsed_time, 4)}ms.')

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
            actor_transform = actor_snapshot._get_transform()
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
