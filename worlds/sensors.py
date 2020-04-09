"""A collection of sensors helpers."""

import math
import collections
import numpy as np
import carla
import time

from functools import wraps
from worlds import utils

# TODO: add support for other sensors: lidar, radar, depth camera etc.


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


class Sensor(object):
    """Base class for sensor wrappers."""
    def __init__(self, parent_actor: carla.Actor, transform=carla.Transform(), attachment_type=None, callback=None,
                 attributes: dict = None):
        self.parent = parent_actor
        self.world = self.parent.get_world()
        self.event_callbacks = [callback] if callback is not None else []  # TODO: migliorare API

        # detector-sensors retrieve data only when triggered (not at each tick!)
        self.sensor, self.is_detector = self._spawn(transform, attachment_type, attributes or dict())

    @property
    def name(self):
        raise NotImplementedError

    def add_callback(self, callback):
        if callback is not None:
            self.event_callbacks.append(callback)

    def clear_callbacks(self):
        self.event_callbacks.clear()

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

    def _spawn(self, transform, attachment_type=None, attributes: dict = None):
        """Spawns itself within a carla.World."""
        if attachment_type is None:
            attachment_type = carla.AttachmentType.Rigid

        sensor_bp: carla.ActorBlueprint = self.world.get_blueprint_library().find(self.name)

        for attr, value in attributes.items():
            if sensor_bp.has_attribute(attr):
                sensor_bp.set_attribute(attr, str(value))
            else:
                print(f'Sensor {self.name} has no attribute `{attr}`')

        sensor_actor = self.world.spawn_actor(sensor_bp, transform, self.parent, attachment_type)
        is_detector = not sensor_bp.has_attribute('sensor_tick')

        return sensor_actor, is_detector

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

    def convert_image(self, image: carla.Image, dtype=np.dtype("uint8")):
        if self.color_converter is not carla.ColorConverter.Raw:
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


class DepthCameraSensor(CameraSensor):
    @property
    def name(self):
        return 'sensor.camera.depth'


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
# -- Sensors specifications
# -------------------------------------------------------------------------------------------------

class SensorSpecs(object):
    ATTACHMENT_TYPE = {'SpringArm': carla.AttachmentType.SpringArm,
                       'Rigid': carla.AttachmentType.Rigid,
                       None: carla.AttachmentType.Rigid}

    COLOR_CONVERTER = {'Raw': carla.ColorConverter.Raw,
                       'CityScapesPalette': carla.ColorConverter.CityScapesPalette,
                       'Depth': carla.ColorConverter.Depth,
                       'LogarithmicDepth': carla.ColorConverter.LogarithmicDepth,
                       None: carla.ColorConverter.Raw}

    @staticmethod
    def _get_position(position: str = None) -> carla.Transform:
        if position == 'top':
            return carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0))
        elif position == 'top-view':
            return carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0))
        elif position == 'front':
            return carla.Transform(carla.Location(x=1.5, z=1.8))
        elif position == 'on-top':
            return carla.Transform(carla.Location(x=-0.9, y=0.0, z=2.2))
        else:
            return carla.Transform()

    @staticmethod
    def camera(kind: str, transform: carla.Transform = None, position: str = None, attachment_type=None,
               color_converter=None, **kwargs) -> dict:
        assert kind in ['rgb', 'depth', 'semantic_segmentation']
        return dict(type='sensor.camera.' + kind,
                    transform=transform or SensorSpecs._get_position(position),
                    attachment_type=SensorSpecs.ATTACHMENT_TYPE[attachment_type],
                    color_converter=SensorSpecs.COLOR_CONVERTER[color_converter],
                    attributes=kwargs)

    @staticmethod
    def rgb_camera(transform: carla.Transform = None, position: str = None, attachment_type='SpringArm',
                   color_converter='Raw', **kwargs):
        return SensorSpecs.camera('rgb', transform, position, attachment_type, color_converter, **kwargs)

    @staticmethod
    def depth_camera(transform: carla.Transform = None, position: str = None, attachment_type='SpringArm',
                     color_converter='LogarithmicDepth', **kwargs):
        return SensorSpecs.camera('depth', transform, position, attachment_type, color_converter, **kwargs)

    @staticmethod
    def segmentation_camera(transform: carla.Transform = None, position: str = None, attachment_type='SpringArm',
                            color_converter='CityScapesPalette', **kwargs):
        return SensorSpecs.camera('semantic_segmentation', transform, position, attachment_type, color_converter, **kwargs)

    @staticmethod
    def detector(kind: str, transform: carla.Transform = None, position: str = None, attachment_type=None,
                 **kwargs) -> dict:
        assert kind in ['collision', 'lane', 'obstacle']
        return dict(type='sensor.other.' + kind,
                    transform=transform or SensorSpecs._get_position(position),
                    attachment_type=SensorSpecs.ATTACHMENT_TYPE[attachment_type],
                    attributes=kwargs)

    @staticmethod
    def collision_detector(transform: carla.Transform = None, position: str = None, attachment_type='Rigid', **kwargs):
        return SensorSpecs.detector('collision', transform, position, attachment_type, **kwargs)

    @staticmethod
    def lane_detector(transform: carla.Transform = None, position: str = None, attachment_type='Rigid', **kwargs):
        return SensorSpecs.detector('lane', transform, position, attachment_type, **kwargs)

    @staticmethod
    def obstacle_detector(transform: carla.Transform = None, position: str = None, attachment_type='Rigid', **kwargs):
        return SensorSpecs.detector('obstacle', transform, position, attachment_type, **kwargs)

    @staticmethod
    def other(kind: str, transform: carla.Transform = None, position: str = None, attachment_type=None, **kwargs) -> dict:
        assert kind in ['imu', 'gnss']  # TODO: add 'lidar' and 'radar'
        return dict(type='sensor.other.' + kind,
                    transform=transform or SensorSpecs._get_position(position),
                    attachment_type=SensorSpecs.ATTACHMENT_TYPE[attachment_type],
                    attributes=kwargs)

    @staticmethod
    def imu(transform: carla.Transform = None, position: str = None, attachment_type='Rigid', **kwargs):
        return SensorSpecs.other('imu', transform, position, attachment_type, **kwargs)

    @staticmethod
    def gnss(transform: carla.Transform = None, position: str = None, attachment_type='Rigid', **kwargs):
        return SensorSpecs.other('imu', transform, position, attachment_type, **kwargs)
