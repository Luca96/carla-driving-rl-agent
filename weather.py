"""
CARLA Dynamic Weather:

Change Sun position smoothly with time and generate storms occasionally.
"""

import sys
import math


def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))


class Sun(object):
    """Sun weather class."""
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = 0.0

    def tick(self, delta_seconds):
        self._t += 0.008 * delta_seconds
        self._t %= 2.0 * math.pi
        self.azimuth += 0.25 * delta_seconds
        self.azimuth %= 360.0
        self.altitude = 35.0 * (math.sin(self._t) + 1.0)

    def __str__(self):
        return 'Sun(%.2f, %.2f)' % (self.azimuth, self.altitude)


class Storm(object):
    """Storm weather class."""
    def __init__(self, precipitation):
        self._t = precipitation if precipitation > 0.0 else -50.0
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.puddles = 0.0
        self.wind = 0.0

    def tick(self, delta_seconds):
        delta = (1.3 if self._increasing else -1.3) * delta_seconds
        self._t = clamp(delta + self._t, -250.0, 100.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 90.0)
        self.rain = clamp(self._t, 0.0, 80.0)
        delay = -10.0 if self._increasing else 90.0
        self.puddles = clamp(self._t + delay, 0.0, 75.0)
        self.wind = clamp(self._t - delay, 0.0, 80.0)
        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False

    def __str__(self):
        return 'Storm(clouds=%d%%, rain=%d%%, wind=%d%%)' % (self.clouds, self.rain, self.wind)


class Weather(object):
    """Dynamic weather wrapper of a CARLA's worlds.get_weather()"""
    def __init__(self, carla_world):
        self.world = carla_world
        self.weather = self.world.weather
        self._sun = Sun(self.weather.sun_azimuth_angle, self.weather.sun_altitude_angle)
        self._storm = Storm(self.weather.precipitation)

    def _tick(self, delta_seconds):
        self._sun.tick(delta_seconds)
        self._storm.tick(delta_seconds)
        self.weather.cloudiness = self._storm.clouds
        self.weather.precipitation = self._storm.rain
        self.weather.precipitation_deposits = self._storm.puddles
        self.weather.wind_intensity = self._storm.wind
        self.weather.sun_azimuth_angle = self._sun.azimuth
        self.weather.sun_altitude_angle = self._sun.altitude

    def start(self, speed=1.0, tick_seconds=30.0, debug=False):
        """Starts simulating dynamic weather conditions.
            @:argument speed: determines how fast the weather gets updated. Default: 1.0
            @:argument tick_seconds: number of seconds for worlds.wait_for_tick(seconds). Default 30.0
            @:argument debug: prints some useful info for debug purpose. Default: False
        """
        # TODO: include in a separate thread as it runs indefinitely!

        update_freq = 0.1 / speed
        elapsed_time = 0.0

        while True:
            timestamp = self.world.wait_for_tick(seconds=tick_seconds).timestamp
            elapsed_time += timestamp.delta_seconds

            if elapsed_time > update_freq:
                self._tick(speed * elapsed_time)
                self.world.set_weather(self.weather)

                if debug:
                    sys.stdout.write('\r' + str(self) + 12 * ' ')
                    sys.stdout.flush()

                elapsed_time = 0.0

    # TODO: Are stop, resume, restart methods necessary?

    def __str__(self):
        return '%s %s' % (self._sun, self._storm)
