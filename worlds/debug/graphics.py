import datetime
import math
import os
import carla
import numpy as np
import pygame

import worlds.utils as utils
import worlds.tools.misc as misc
from worlds.navigation import Route

from worlds.sensors import Sensor, IMUSensor
from worlds.tools.misc import compute_magnitude_angle


class DebugInfo(object):
    """Debug environments by showing useful information"""
    default_font = 'ubuntumono'

    def __init__(self, width: int, height: int, notification_bar_height=40):
        self.dim = (width, height)

        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]

        mono = self.default_font if self.default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)

        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, notification_bar_height),
                                         (0, height - notification_bar_height))
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, snapshot: carla.WorldSnapshot):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = snapshot.frame
        self.simulation_time = snapshot.elapsed_seconds

    def tick(self, clock: pygame.time.Clock):
        self._notifications.tick(clock)

        if not self._show_info:
            return

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time))
        ]

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            # info_surface = pygame.Surface((220, self.dim[1]))
            info_surface = pygame.Surface((self.dim[0] // 3.2, self.dim[1]))
            info_surface.set_alpha(100)

            display.blit(info_surface, (0, 0))
            v_offset = 4

            bar_h_offset = 100
            bar_width = 106

            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break

                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)

                    item = None
                    v_offset += 18

                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])

                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))

                        pygame.draw.rect(display, (255, 255, 255), rect)

                    item = item[0]

                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))

                v_offset += 18

        self._notifications.render(display)


# -------------------------------------------------------------------------------------------------
# -- HUD
# -------------------------------------------------------------------------------------------------

class HUD(DebugInfo):
    """HUD: provides a pygame window that shows what the vehicle sees, some statistics, etc."""
    def __init__(self, world, width: int, height: int, **kwargs):
        assert world.__class__.__name__ == 'World'
        super().__init__(width, height, **kwargs)
        self.world = world

    def tick(self, clock):
        super().tick(clock)
        world = self.world

        t = world.player.get_transform()
        c = world.player.get_control()
        v = world.player.get_velocity()

        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''

        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        similarity = utils.cosine_similarity(world.player.get_transform().get_forward_vector(),
                                             world.route.closest_path.waypoint.transform.get_forward_vector())

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            # '',
            'Vehicle: % 20s' % utils.get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            # '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % world.imu_sensor.accelerometer,
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % world.imu_sensor.gyroscope,
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            # 'Height:  % 18.0f m' % t.location.z,
            'Sim: %.2f, limit: %d km/h' % (similarity, world.player.get_speed_limit()),
            'VehicleFeatures: ' + str(np.round(world.get_vehicle_features(), 1)),
            'RoadFeatures: ' + str(np.round(world.get_road_features(), 1)),
            '']

        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]

        self._info_text += [
            '',
            'Collision: ' + str(world.num_collisions),
            collision,
            '',
            'Number of vehicles: % 5d' % len(vehicles),
            # '-- Reward Signal --',
            'Elapsed time: %12s' % datetime.timedelta(seconds=int(world.elapsed_time)),
            'Distance travelled: %5d / %d' % (int(world.travelled_distance), int(world.route.size)),
            world.debug_reward(),

            # TODO: does not take into account when vehicle is OUTSIDE the road!
            world.waypoint_info(),
            world.lane_info(),
            'Distance (i: %d, d: %.2fm)' % (world.route.closest_path.index, world.route.closest_path.distance),
            'Destination: %.2fm' % world.distance_to_destination()
        ]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

            distance = lambda l: math.sqrt(
                (l.x - t.location.x) ** 2 + (l.y - t.location.y) ** 2 + (l.z - t.location.z) ** 2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]

            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break

                vehicle_type = utils.get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))


class CARLADebugInfo(DebugInfo):
    """Provides debugging facilities for SynchronousCARLAEnvironment"""
    def __init__(self, width: int, height: int, environment, **kwargs):
        # TODO: ugly check: cannot import class due to circular reference :(
        assert environment.__class__.__name__ == 'SynchronousCARLAEnvironment'
        super().__init__(width, height, **kwargs)
        self.environment = environment

    def tick(self, clock: pygame.time.Clock):
        super().tick(clock)
        env = self.environment

        vehicle = env.vehicle
        world: carla.World = env.world
        route: Route = env.route
        imu_sensor: IMUSensor = env.sensors['sensor.other.imu']

        t = vehicle.get_transform()
        c = vehicle.get_control()

        compass = imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''

        # colhist = world.collision_sensor.get_collision_history()
        # collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        collision = [1 for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.get_actors().filter('vehicle.*')

        similarity = utils.cosine_similarity(vehicle.get_transform().get_forward_vector(),
                                             route.closest_path.waypoint.transform.get_forward_vector())

        self._info_text += [
            '',
            'Vehicle: % 20s' % utils.get_actor_display_name(vehicle, truncate=20),
            'Map:     % 20s' % world.get_map().name,
            '',
            'Speed:   % 15.0f km/h' % utils.speed(vehicle),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % imu_sensor.accelerometer,
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % imu_sensor.gyroscope,
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            # 'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            'Sim: %.2f, limit: %d km/h' % (similarity, vehicle.get_speed_limit()),
            ''
        ]

        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]

        self._info_text += [
            '',
            'Collision: ' + str(env.num_collisions),
            collision,
            '',
            'Number of vehicles: % 5d' % len(vehicles),
            # '-- Reward Signal --',
            'Distance travelled: %5d / %d' % (int(env.travelled_distance), int(route.size)),
            # world.debug_reward(),
            f'Reward: {round(env.reward(), 2)}',
            f'Skill: {env.get_skill_name()}, action_penalty: {env._action_penalty(env.prev_actions)}',

            # TODO: does not take into account when vehicle is OUTSIDE the road!
            # world.waypoint_info(),
            # world.lane_info(),
            'Distance (i: %d, d: %.2fm)' % (route.closest_path.index, route.closest_path.distance),
            'Destination: %.2fm' % route.distance_to_destination()
        ]


class CARLADebugInfoSmall(DebugInfo):
    """Provides debugging facilities for SynchronousCARLAEnvironment"""
    def __init__(self, width: int, height: int, environment, **kwargs):
        # TODO: ugly check: cannot import class due to circular reference :(
        assert environment.__class__.__name__ == 'SynchronousCARLAEnvironment'
        super().__init__(width, height, **kwargs)
        self.environment = environment

    def tick(self, clock: pygame.time.Clock):
        env = self.environment

        vehicle = env.vehicle
        world: carla.World = env.world

        c = vehicle.get_control()

        self._info_text = [
            'FPS % 20d' % clock.get_fps(),
            'Speed:   % 15.0f km/h' % utils.speed(vehicle),
            '',
            'Throttle: %.2f' % c.throttle,
            'Steer: %.2f' % c.steer,
            'Brake: %.2f' % c.brake,
            ('Reverse:', c.reverse),
            ('Hand brake:', c.hand_brake),
            'Gear: %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear),
            '',
            'Reward % 20.2f' % env.reward(),
            'Collision: %d' % env.num_collisions,
            'Skill: %s' % env.get_skill_name(),
            f'action_penalty: %.2f' % env._action_penalty(env.prev_actions)
        ]


# -------------------------------------------------------------------------------------------------
# -- Fading Text
# -------------------------------------------------------------------------------------------------

class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)
