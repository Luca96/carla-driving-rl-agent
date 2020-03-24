"""A collection of sensors helpers."""

import numpy as np
import carla

from worlds import utils

import math
import collections

# TODO: add support for other sensors: lidar, radar, depth camera etc.


class Sensor(object):
    """Base class for sensor wrappers."""
    def __init__(self, parent_actor: carla.Actor, transform=carla.Transform(), attachment_type=None, callback=None):
        self.parent = parent_actor
        self.world = self.parent.get_world()
        self.event_callbacks = [callback] if callback is not None else []  # TODO: migliorare API

        self.sensor = self._spawn(transform, attachment_type)
        # self.sensor.listen(self.on_event)

    @property
    def name(self):
        raise NotImplementedError

    def add_callback(self, callback):
        if callback is not None:
            self.event_callbacks.append(callback)

    @staticmethod
    def create(sensor_name, **kwargs):
        if sensor_name in ['collision', 'collision_detector', 'collision_sensor']:
            return CollisionSensor(**kwargs)

        elif sensor_name in ['lane', 'lane_sensor', 'lane_invasion_sensor']:
            return LaneInvasionSensor(**kwargs)

        elif sensor_name in ['gnss', 'gnss_sensor']:
            return GnssSensor(**kwargs)

        elif sensor_name in ['imu', 'imu_sensor']:
            return IMUSensor(**kwargs)

        elif sensor_name in ['camera', 'camera.rgb', 'camera_sensor', 'rgb_camera']:
            return RGBCameraSensor(**kwargs)

        elif sensor_name in ['semantic_camera', 'semantic_segmentation_camera', 'semantic_camera_sensor']:
            return SemanticCameraSensor(**kwargs)

        else:
            raise ValueError(f'Cannot create sensor `{sensor_name}`')

    def start(self):
        """Start listening for events"""
        if not self.sensor.is_listening:
            self.sensor.listen(self.on_event)
        else:
            print(f'Sensor {self.name} is already been started!')

    def stop(self):
        """Stop listening for events"""
        self.sensor.stop()

    def _spawn(self, transform, attachment_type=None,):
        """Spawns itself within a carla.World."""
        if attachment_type is None:
            attachment_type = carla.AttachmentType.Rigid

        sensor_bp: carla.ActorBlueprint = self.world.get_blueprint_library().find(self.name)

        if sensor_bp.has_attribute('sensor_tick'):
            pass
            # sensor_bp.set_attribute('sensor_tick', '0.033')
        else:
            print(f'Sensor {self.name} has no attribute `sensor_tick`')

        sensor_actor = self.world.spawn_actor(sensor_bp, transform, self.parent, attachment_type)
        # sensor_actor = self.world.spawn_actor(sensor_bp, transform,
        #                                       attach_to=self.parent,
        #                                       attachment=attachment_type)
        return sensor_actor

    def on_event(self, event):
        for callback in self.event_callbacks:
            callback(event)

    def destroy(self):
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None

        self.parent = None
        self.world = None


# -------------------------------------------------------------------------------------------------
# -- Camera Sensors
# -------------------------------------------------------------------------------------------------

class CameraSensor(Sensor):
    def __init__(self, color_converter=carla.ColorConverter.Raw, **kwargs):
        super().__init__(**kwargs)
        self.color_converter = color_converter

    @property
    def name(self):
        raise NotImplementedError

    # def on_event(self, event):
    #     super().on_event(event)

    def convert_image(self, image: carla.Image, dtype=np.dtype("uint8")):
        image.convert(self.color_converter)
        array = np.frombuffer(image.raw_data, dtype=dtype)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array


class RGBCameraSensor(CameraSensor):
    @property
    def name(self):
        return 'sensor.camera.rgb'


class SemanticCameraSensor(CameraSensor):
    @property
    def name(self):
        return 'sensor.camera.semantic_segmentation'


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
        print('Collision with %r' % actor_type)

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

        # TODO: uncomment
        # self.hud.notification('Crossed line %s' % ' and '.join(text))

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
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
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
#
# class CameraSensor(object):
#     """Wraps a single camera sensor for ease of use within a CameraManager"""
#     # TODO: use logger insted of print for warning statement
#
#     def __init__(self, kind: str, name: str, color_converter=carla.ColorConverter.Raw):
#         """@:arg kind: one of 'rgb', 'depth', 'semantic_segmentation'."""
#         # TODO: add LIDAR and RADAR support?
#         self.kind = kind
#         self.name = name
#         self.color_converter = color_converter
#         self.blueprint = None
#         self.callback = None
#         self.actor = None
#
#     def load_blueprint(self, blueprint_library):
#         """Uses the blueprint_library to find itself according to self.kind"""
#         self.blueprint = blueprint_library.find('sensor.camera.' + self.kind)
#
#     def spawn_actor(self, world: carla.World, transform: carla.Transform, attach_to: carla.Actor, attachment_type: carla.AttachmentType):
#         """Spawns itself within a carla.World."""
#         if self.blueprint is None:
#             raise ValueError('Blueprint is None, call "load_blueprint()" before "spawn_actor"!')
#
#         if self.actor is not None:
#             print('[Warning] CameraSensor.spawn_actor: actor not None, destroying it before been spawn!')
#             self.actor.destroy()
#
#         self.actor = world.spawn_actor(self.blueprint, transform, attach_to, attachment_type)
#
#     def set_attributes(self, **kwargs):
#         for key, value in kwargs.items():
#             if self.blueprint.has_attribute(key):
#                 self.blueprint.set_attribute(key, value)
#             else:
#                 print(f'[Warning] CameraSensor.set_attributes: attribute "{key}" is not available for ' +
#                       f'sensor {self.kind}!')
#
#     def listen(self, callback):
#         pass
#
#     def destroy(self):
#         if self.actor is not None:
#             self.actor.destroy()
#             self.actor = None
#
#         self.blueprint = None
#         self.callback = None
#
#
# class CombinedCameraSensor(CameraSensor):
#     """Wraps multiple sensors as a single one."""
#     pass
