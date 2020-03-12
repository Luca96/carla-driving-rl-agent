import sys
import math
import random
import carla

import worlds.debug
import worlds.utils as utils

from worlds.utils import WAYPOINT_DICT
from worlds.navigation import Route, RoutePlanner
from worlds.managers import CameraManager
from worlds.sensors import CollisionSensor, LaneInvasionSensor, GnssSensor, IMUSensor


class World(object):
    """A high-level wrapper of a carla.World instance."""

    def __init__(self, carla_world, hud, actor_filter, map='Town1', route_resolution=2.0, init=True):
        # TODO: implement loading a specific map.
        print('### WORLD::INIT ###')
        self.world = carla_world
        self.actor_role_name = 'hero'
        self.map = self.world.get_map()

        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713

        # variables for reward computation
        self.elapsed_time = 0.0
        self.travelled_distance = 0.0
        self.num_collisions = 0
        self.last_location = None

        # Route planning:
        self.route = Route(planner=RoutePlanner(self.map, sampling_resolution=route_resolution))
        self.initial_position: carla.Transform = None  # initial spawn position
        self.target_position: carla.Transform = None  # final destination

        # Debug vehicle trail
        self.debug = self.world.debug
        self.lane_type = carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk
        self.current_waypoint = None
        self.next_waypoint = None

        self.hud = hud
        self.player = None

        # Sensors
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.camera_manager = None
        self.sensor_image = None

        # Weather
        self._weather_presets = utils.find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter

        if init:
            self.start()

        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def reward(self, a=-1, b=-1000, c=1, d=-0.8):
        """Returns a scalar reward."""
        # TODO: include the distance from vehicle to closest (or next) route waypoint.
        # TODO: include a penalty for law compliance: exceeding the speed limit, red traffic light...

        # Time term: go to destination as fast as possible
        spent_time = a * self.elapsed_time

        # Collision term: do as few collision as possible
        collision_penalty = b * self.num_collisions

        # Efficiency term: 'positive' if travelled_distance < route_size, 'negative' otherwise
        if self.travelled_distance <= self.route.size:
            efficiency = self.travelled_distance
        else:
            efficiency = self.route.size - self.travelled_distance

        # Destination term: signal how far the target location is
        destination = d * self.distance_to_destination()

        return spent_time + collision_penalty + c * efficiency + destination

    def start(self):
        print('> worlds.start')
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0

        blueprint = self._get_random_blueprint()

        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0

            self.destroy()
            self._try_spawn_player(blueprint, spawn_point)

        while self.player is None:
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()

            self._try_spawn_player(blueprint, spawn_point)

        # Pick a random target destination
        while self.target_position is None:
            point = random.choice(self.map.get_spawn_points())

            if point != self.initial_position:
                self.target_position = point

        self.route.plan(origin=self.initial_position.location,
                        destination=self.target_position.location)

        # Variables for reward computation
        self.elapsed_time = 0.0
        self.travelled_distance = 0.0
        self.num_collisions = 0
        print('target-location', self.target_position.location)

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud, callback=self.on_collision)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)

        self.camera_manager = CameraManager(self.player, self.hud, callback=self.on_image)
        self.camera_manager.transform_index = cam_pos_index
        # self.camera_manager.set_sensor(cam_index, force_respawn=True, notify=False)
        self.camera_manager.set_sensor(cam_index, notify=False)
        self.sensor_image = None

        actor_type = utils.get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def restart(self):
        print('--> world.restart')
        self.destroy()
        self.start()

    def apply_control(self, control: carla.VehicleControl):
        self.player.apply_control(control)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)

        preset = self._weather_presets[self._weather_index]

        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def waypoint_info(self):
        """Returns a string about the current waypoint."""
        w = self.current_waypoint

        if w is None:
            return 'Waypoint: None'

        return f'Waypoint (intersection: {w.is_intersection}, junction: {w.is_junction})'

    def lane_info(self):
        """Returns a string about the current waypoint's lanes."""
        w = self.current_waypoint

        if w is None:
            return 'Lane: None'

        return f'Lane (change: {w.lane_change}, type: {w.lane_type}, width: {int(w.lane_width)}, ' + \
               f'left: {w.left_lane_marking.type}, right: {w.right_lane_marking.type})'

    def get_vehicle_features(self):
        """Returns a dict(speed, position, destination, compass) representing the vehicle location state"""
        t = self.player.get_transform()
        v = self.player.get_velocity()

        return [min(3.6 * utils.vector_norm(v), 150.0),
                t.location.x,
                t.location.y,
                self.target_position.location.x,
                self.target_position.location.y,
                math.radians(self.imu_sensor.compass)]

    def get_road_features(self):
        waypoint = self.map.get_waypoint(self.player.get_location())

        return [waypoint.is_intersection,
                waypoint.is_junction,
                waypoint.lane_width,
                WAYPOINT_DICT['lane_type'][waypoint.lane_type],
                WAYPOINT_DICT['lane_change'][waypoint.lane_change],
                WAYPOINT_DICT['lane_marking_type'][waypoint.left_lane_marking.type],
                WAYPOINT_DICT['lane_marking_type'][waypoint.right_lane_marking.type]]

    def on_image(self, image):
        if image is not None:
            self.sensor_image = image

    def distance_to_destination(self):
        """Returns the distance from the current vehicle location to the destination (i.e. target location)."""
        return self.route.distance_to_destination()

    def on_collision(self, event):
        """Just increase the number of collisions. The next 'tick' will erase this number."""
        actor = event.other_actor
        # TODO: discover the other_actor's type, if is human terminate the episode, if is a vehicle add a bigger penalty
        self.num_collisions += 1

    def tick(self, clock, debug=True):
        milliseconds = clock.get_time() / 1000
        self.elapsed_time += milliseconds
        self._update_distance()
        self._update_waypoints()
        self.route.update_closest_waypoint(location=self.player.get_location())

        reward = self.reward()

        self.hud.tick(self, clock)

        if debug:
            # self.debug_vehicle_trail(life_time=milliseconds * 10)
            self.route.draw_closest_waypoint(self.world.debug, self.player.get_location(), life_time=milliseconds)
            worlds.debug.draw_route(self.debug, self.route.path, life_time=milliseconds * 2)

        # Clear collisions
        # self.num_collisions = 0

        return reward

    def render(self, display):
        # print('worlds.render', self)
        self.camera_manager.render(display)
        self.hud.render(display)

        return self.sensor_image

    def destroy_sensors(self):
        print('worlds.destroy_sensors')
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        print('> worlds.destroy')

        actors = [
            self.camera_manager,
            self.collision_sensor,
            self.lane_invasion_sensor,
            self.gnss_sensor,
            self.imu_sensor,
            self.player
        ]

        for actor in actors:
            if actor is not None:
                actor.destroy()

        self.camera_manager = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.player = None
        self.sensor_image = None

    # -- Private methods --------------------------------------------------------------------------

    def _get_random_blueprint(self):
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', self.actor_role_name)

        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)

        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')

        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])
        else:
            print("No recommended values for 'speed' attribute")

        return blueprint

    def _try_spawn_player(self, blueprint, spawn_point):
        self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        self.initial_position = spawn_point
        self.last_location = spawn_point.location

        self.current_waypoint = self.map.get_waypoint(self.last_location, lane_type=self.lane_type)
        self.next_waypoint = self.current_waypoint

    def _update_waypoints(self):
        self.current_waypoint = self.next_waypoint
        self.next_waypoint = self.map.get_waypoint(self.player.get_location(), lane_type=self.lane_type)

    def _update_distance(self, distance=utils.l2_norm):
        # distance, method 1
        l1, l2 = self.last_location, self.player.get_location()

        # TODO: L2-distance is an approximation, use speed!
        self.travelled_distance += distance(l1, l2)  # meters
        self.last_location = l2

        # distance, method 2
        # velocity = self.player.get_velocity()
        # speed = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)  # km/h
        # self.distance += speed * clock.get_time() / 1e6  # kilometers
