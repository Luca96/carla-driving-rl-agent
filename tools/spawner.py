"""Spawn NPCs into the simulation using the Traffic Manager interface"""

import sys
import time
import random
import logging
import carla

# from typing import List, Tuple, Dict
# Point = Tuple[int, int]
SpawnActor = carla.command.SpawnActor
DestroyActor = carla.command.DestroyActor


class Spawner(object):
    """Util class for spawning NPCs like vehicles, walkers, etc."""

    def __init__(self, carla_client):
        self.client = carla_client
        self.world = self.client.get_world()

        self.blueprint_lib = self.world.get_blueprint_library()
        self.car_blueprints = self.__get_car_blueprints()
        self.walker_blueprints = self.blueprint_lib.filter('walker.pedestrian.*')
        self.walker_controller = self.blueprint_lib.find('controller.ai.walker')
        self.spawn_points = random.shuffle(self.world.get_map().get_spawn_points())  # randomize spawn points

        self.vehicles = []
        self.walkers = []
        self.ids = []  # keep the IDs of each spawned NPC

    def spawn_vehicles(self, num: int):
        """Spawn vehicles (car, bikes, ...) into the current CARLA's worlds.
            @:argument num: number of vehicles to try to spawn.
        """
        spawn_points = self.__get_vehicle_spawn_points(num)

        for transform in spawn_points:
            car_blueprint = random.choice(self.car_blueprints)
            car_blueprint.set_attrinute('role_name', 'autopilot')

            if car_blueprint.has_attribute('color'):
                color = random.choice(car_blueprint.get_attribute('color').recommended_values)
                car_blueprint.set_attrinute('color', color)

            if car_blueprint.has_attribute('driver_id'):
                driver_id = random.choice(car_blueprint.get_attribute('driver_id').recommended_values)
                car_blueprint.set_attrinute('driver_id', driver_id)

            # spawn blueprint into the worlds
            vehicle = self.world.try_spawn_actor(car_blueprint, transform)
            self.vehicles.append(vehicle)

        print(f'Spawned {len(self.vehicles)} vehicles.')

        time.sleep(1)
        for vehicle in self.vehicles:
            vehicle.set_autopilot(True)

    def spawn_walkers(self, num: int, run_factor=0.0, cross_factor=0.0):
        """Spawn walkers into the current CARLA's worlds.
            @:argument num: number of walkers to try to spawn.
            @:argument run_factor: the percentage of walkers that should be run.
            @:argument cross_factor: the percentage of walkers that should cross streets.
        """
        # TODO: can be invoked only one time, fix!
        assert len(self.walkers) == 0

        spawn_points = self.__get_walker_spawn_points(num)
        batch = []
        walker_speed = []

        # 1. spawn the walker object
        for spawn_point in spawn_points:
            walker_bp = random.choice(self.walker_blueprints)

            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')

            if walker_bp.has_attribute('speed'):
                if random.random() > run_factor:
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])  # walk
                else:
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])  # run
            else:
                # no speed
                walker_speed.append(0.0)

            batch.append(SpawnActor(walker_bp, spawn_point))

        results = self.client.apply_batch_sync(batch, True)

        # find which walker has been correctly spawned
        for i, result in enumerate(results):
            if result.error:
                logging.error(result.error)
            else:
                self.walkers.append(dict(id=result.actor_id, speed=walker_speed[i]))

        # 2. spawn the walker controller
        batch = [SpawnActor(self.walker_controller, carla.Transform(), walker['id']) for walker in self.walkers]
        results = self.client.apply_batch_sync(batch, True)

        for i, result in enumerate(results):
            if result.error:
                logging.error(result.error)
            else:
                self.walkers[i]['con'] = result.actor_id

        # 4. Put together walkers and controllers to get the objects from their id
        for walker in self.walkers:
            self.ids.append(walker['con'])
            self.ids.append(walker['id'])

        all_actors = self.world.get_actors(self.ids)
        self.world.wait_for_tick()

        # 5. initialize each controller and set target to walk to
        self.world.set_pedestrians_cross_factor(cross_factor)

        for i in range(0, len(self.ids), 2):
            all_actors[i].start()  # start walker
            all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            all_actors[i].set_max_speed(float(self.walkers[i // 2]['speed']))

        print(f'Spawned {len(self.walkers)} walkers.')

    def destroy_all(self):
        """Destroys all spawned vehicles and walkers."""
        print(f'Destroying {len(self.vehicles)} vehicles.\n')
        self.client.apply_batch([DestroyActor(x) for x in self.vehicles])

        # stop walker controllers
        all_actors = self.world.get_actors(self.ids)
        for i in range(0, len(self.ids), 2):
            all_actors[i].stop()

        print(f'\ndestroying {len(self.walkers)} walkers')
        self.client.apply_batch([DestroyActor(x) for x in self.ids])

    def __get_vehicle_spawn_points(self, num: int):
        num_points = min(num, len(self.spawn_points))
        spawn_points = [self.spawn_points.pop() for _ in range(num_points)]
        return spawn_points

    def __get_walker_spawn_points(self, num: int):
        spawn_points = []
        for i in range(num):
            point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()

            if loc is not None:
                point.location = loc
                spawn_points.append(point)

        return spawn_points

    def __get_car_blueprints(self):
        blueprints = self.blueprint_lib.filter('vehicle.*')
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]  # TODO: this prevents spawning (motor)bikes?
        blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
        return blueprints
