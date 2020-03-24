#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import carla
import queue


class CARLASyncContext(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, sensors: dict, fps=30, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / fps
        self._settings = None

        # Make a queue for each sensor and for world:
        self._queues = dict()
        self._add_queue('world', self.world.on_tick)

        for name, sensor in self.sensors.items():
            self._add_queue(name, sensor.add_callback)

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        for sensor in self.sensors.values():
            sensor.start()

        return self

    def tick(self, timeout):
        self.frame = self.world.tick()

        data = dict()
        for name, q in self._queues.items():
            data[name] = self._retrieve_data(q, timeout)

        return data

    def __exit__(self, *args, **kwargs):
        # self.world.apply_settings(self._settings)
        self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=False))

        for sensor in self.sensors.values():
            sensor.stop()

    def _add_queue(self, name, register_event):
        """Registers an even on its own queue identified by name"""
        q = queue.Queue()
        register_event(q.put)
        self._queues[name] = q

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)

            # TODO: sensors that don't tick at each timestep should be handled properly to avoid exceptions
            if data.frame == self.frame:
                return data
