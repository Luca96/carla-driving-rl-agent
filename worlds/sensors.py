"""A collection of sensors helpers."""

import carla

from worlds import utils

import math
import collections


class Sensor(object):
    """Base class for sensor wrappers."""
    def __init__(self, parent_actor: carla.Actor, transform=carla.Transform(), callback=None):
        self.parent = parent_actor
        self.world = self.parent.get_world()
        self.on_event_callback = callback

        self.sensor = self._spawn(transform)
        self.sensor.listen(self.on_event)

    @property
    def name(self):
        raise NotImplementedError

    def _spawn(self, transform):
        """Spawns itself within a carla.World."""
        sensor_bp = self.world.get_blueprint_library().find(self.name)
        sensor_actor = self.world.spawn_actor(sensor_bp, transform, attach_to=self.parent)
        return sensor_actor

    def on_event(self, event):
        if self.on_event_callback is not None:
            self.on_event_callback(event)

    def destroy(self):
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None

        self.parent = None
        self.world = None


# -------------------------------------------------------------------------------------------------
# -- Collision Sensor
# -------------------------------------------------------------------------------------------------

class CollisionSensor(Sensor):
    def __init__(self, parent_actor, hud, callback=None, history_size=4000):
        super().__init__(parent_actor, callback=callback)
        self.history = []
        self.history_size = history_size
        self.hud = hud

    @property
    def name(self):
        return 'sensor.other.collision'

    def get_collision_history(self):
        history = collections.defaultdict(int)

        for frame, intensity in self.history:
            history[frame] += intensity

        return history

    def on_event(self, event):
        super().on_event(event)

        actor_type = utils.get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)

        impulse = event.normal_impulse
        intensity = utils.vector_norm(impulse)
        self.history.append((event.frame, intensity))

        if len(self.history) > self.history_size:
            self.history.pop(0)

    def destroy(self):
        super().destroy()
        self.history.clear()
        self.history = None
        self.history_size = None
        self.hud = None


# -------------------------------------------------------------------------------------------------
# -- Lane-Invasion Sensor
# -------------------------------------------------------------------------------------------------

class LaneInvasionSensor(Sensor):
    def __init__(self, parent_actor, hud, callback=None):
        super().__init__(parent_actor, callback=callback)
        self.hud = hud

    @property
    def name(self):
        return 'sensor.other.lane_invasion'

    def on_event(self, event):
        super().on_event(event)

        # Notify lane invasion
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]

        self.hud.notification('Crossed line %s' % ' and '.join(text))

    def destroy(self):
        super().destroy()
        self.hud = None


# -------------------------------------------------------------------------------------------------
# -- GNSS Sensor
# -------------------------------------------------------------------------------------------------

class GnssSensor(Sensor):
    def __init__(self, parent_actor, callback=None):
        super().__init__(parent_actor, transform=carla.Transform(carla.Location(x=1.0, z=2.8)), callback=callback)
        self.lat = 0.0
        self.lon = 0.0

    @property
    def name(self):
        return 'sensor.other.gnss'

    def on_event(self, event):
        super().on_event(event)
        self.lat = event.latitude
        self.lon = event.longitude

    def destroy(self):
        super().destroy()
        self.lat = None
        self.lon = None


# -------------------------------------------------------------------------------------------------
# -- IMU Sensor
# -------------------------------------------------------------------------------------------------

class IMUSensor(Sensor):
    def __init__(self, parent_actor, callback=None):
        super().__init__(parent_actor)
        self.accelerometer = ()
        self.gyroscope = ()
        self.compass = 0.0

    @property
    def name(self):
        return 'sensor.other.imu'

    def on_event(self, event):
        super().on_event(event)
        limits = (-99.9, 99.9)

        self.accelerometer = (
            max(limits[0], min(limits[1], event.accelerometer.x)),
            max(limits[0], min(limits[1], event.accelerometer.y)),
            max(limits[0], min(limits[1], event.accelerometer.z)))

        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(event.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(event.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(event.gyroscope.z))))

        self.compass = math.degrees(event.compass)

    def destroy(self):
        super().destroy()
        self.accelerometer = None
        self.gyroscope = None
        self.compass = None


# -------------------------------------------------------------------------------------------------
# -- Camera Sensor(s) Wrapper
# -------------------------------------------------------------------------------------------------

class CameraSensor(object):
    """Wraps a single camera sensor for ease of use within a CameraManager"""
    # TODO: use logger insted of print for warning statement

    def __init__(self, kind: str, name: str, color_converter=carla.ColorConverter.Raw):
        """@:arg kind: one of 'rgb', 'depth', 'semantic_segmentation'."""
        # TODO: add LIDAR and RADAR support?
        self.kind = kind
        self.name = name
        self.color_converter = color_converter
        self.blueprint = None
        self.callback = None
        self.actor = None

    def load_blueprint(self, blueprint_library):
        """Uses the blueprint_library to find itself according to self.kind"""
        self.blueprint = blueprint_library.find('sensor.camera.' + self.kind)

    def spawn_actor(self, world: carla.World, transform: carla.Transform, attach_to: carla.Actor, attachment_type: carla.AttachmentType):
        """Spawns itself within a carla.World."""
        if self.blueprint is None:
            raise ValueError('Blueprint is None, call "load_blueprint()" before "spawn_actor"!')

        if self.actor is not None:
            print('[Warning] CameraSensor.spawn_actor: actor not None, destroying it before been spawn!')
            self.actor.destroy()

        self.actor = world.spawn_actor(self.blueprint, transform, attach_to, attachment_type)

    def set_attributes(self, **kwargs):
        for key, value in kwargs.items():
            if self.blueprint.has_attribute(key):
                self.blueprint.set_attribute(key, value)
            else:
                print(f'[Warning] CameraSensor.set_attributes: attribute "{key}" is not available for ' +
                      f'sensor {self.kind}!')

    def listen(self, callback):
        pass

    def destroy(self):
        if self.actor is not None:
            self.actor.destroy()
            self.actor = None

        self.blueprint = None
        self.callback = None


class CombinedCameraSensor(CameraSensor):
    """Wraps multiple sensors as a single one."""
    pass
