"""Utility functions for environment.py"""

import os
import cv2
import time
import random
import numpy as np
import carla
import pygame
import threading

from functools import wraps


# def load_nasnet_mobile(folder='weights/keras', weights='nasnet_mobile.h5'):
#     path = os.path.join(folder, weights)
#     return tf.keras.applications.NASNetMobile(weights=path, pooling='avg')
#
#
# def get_image_compressor_model(backend='nasnet_mobile', **kwargs):
#     if backend == 'nasnet_mobile':
#         base_model = load_nasnet_mobile(**kwargs)
#     else:
#         raise ValueError(f'backend: {backend} is not available!')
#
#     pool2d = base_model.get_layer('global_average_pooling2d')
#
#     return tf.keras.models.Model(inputs=base_model.input,
#                                  outputs=pool2d.output,
#                                  name=f'{backend}-ImageCompressor')

# source: https://stackoverflow.com/questions/3620943/measuring-elapsed-time-with-the-time-module/46544199
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


def get_client(address, port, timeout=2.0) -> carla.Client:
    """Connects to the simulator.
        @:returns a carla.Client instance if the CARLA simulator accepts the connection.
    """
    client: carla.Client = carla.Client(address, port)
    client.set_timeout(timeout)
    return client


def get_display(window_size, mode=pygame.HWSURFACE | pygame.DOUBLEBUF):
    """Returns a display used to render images and text.
        :param window_size: a tuple (width: int, height: int)
        :param mode: pygame rendering mode. Default: pygame.HWSURFACE | pygame.DOUBLEBUF
        :return: a pygame.display instance.
    """
    return pygame.display.set_mode(window_size, mode)


def get_font(size=14):
    return pygame.font.Font(pygame.font.get_default_font(), size)


def display_image(display, image, window_size=(800, 600), blend=False):
    """Displays the given image on a pygame window
    :param blend: whether to blend or not the given image.
    :param window_size: the size of the pygame's window. Default is (800, 600)
    :param display: pygame.display
    :param image: the image (numpy.array) to display/render on.
    """
    # Resize image if necessary
    if (image.shape[1], image.shape[0]) != window_size:
        image = resize(image, size=window_size)

    image_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))

    if blend:
        image_surface.set_alpha(100)

    display.blit(image_surface, (0, 0))


def display_text(display, font, text: [str], color=(255, 255, 255), origin=(0, 0), offset=(0, 2)):
    position = origin

    for line in text:
        display.blit(font.render(line, True, color), position)
        position = (position[0] + offset[0], position[1] + offset[1])


def random_blueprint(world: carla.World, actor_filter='vehicle.*', role_name='agent') -> carla.ActorBlueprint:
    """Retrieves a random blueprint.
        :param world: a carla.World instance.
        :param actor_filter: a string used to filter (select) blueprints. Default: 'vehicle.*'
        :param role_name: blueprint's role_name, Default: 'agent'.
        :return: a carla.ActorBlueprint instance.
    """
    blueprints = world.get_blueprint_library().filter(actor_filter)
    blueprint: carla.ActorBlueprint = random.choice(blueprints)
    blueprint.set_attribute('role_name', role_name)

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
        float(blueprint.get_attribute('speed').recommended_values[1])
        float(blueprint.get_attribute('speed').recommended_values[2])
    else:
        print("No recommended values for 'speed' attribute")

    return blueprint


def random_spawn_point(world_map: carla.Map, different_from: carla.Location = None) -> carla.Transform:
    """Returns a random spawning location.
        :param world_map: a carla.Map instance obtained by calling world.get_map()
        :param different_from: ensures that the location of the random spawn point is different from the one specified here.
        :return: a carla.Transform instance.
    """
    available_spawn_points = world_map.get_spawn_points()

    if different_from is not None:
        while True:
            spawn_point = random.choice(available_spawn_points)

            if spawn_point.location != different_from:
                return spawn_point
    else:
        return random.choice(available_spawn_points)


def spawn_actor(world: carla.World, blueprint: carla.ActorBlueprint, spawn_point: carla.Transform,
                attach_to: carla.Actor = None, attachment_type=carla.AttachmentType.Rigid) -> carla.Actor:
    """Tries to spawn an actor in a CARLA simulator.
        :param world: a carla.World instance.
        :param blueprint: specifies which actor has to be spawned.
        :param spawn_point: where to spawn the actor. A transform specifies the location and rotation.
        :param attach_to: whether the spawned actor has to be attached (linked) to another one.
        :param attachment_type: the kind of the attachment. Can be 'Rigid' or 'SpringArm'.
        :return: a carla.Actor instance.
    """
    # actor = world.try_spawn_actor(blueprint, spawn_point, attach_to=attach_to, attachment=attachment_type)
    actor = world.try_spawn_actor(blueprint, spawn_point, attach_to, attachment_type)

    if actor is None:
        raise ValueError(f'Cannot spawn actor. Try changing the spawn_point ({spawn_point}) to something else.')

    return actor


def resize(image, size: (int, int), interpolation=cv2.INTER_CUBIC):
    """Resize the given image.
        :param image: a numpy array with shape (height, width, channels).
        :param size: (width, height) to resize the image to.
        :param interpolation: Default: cv2.INTER_CUBIC.
        :return: the reshaped image.
    """
    return cv2.resize(image, dsize=size, interpolation=interpolation)


def scale(num, from_interval=(-1.0, +1.0), to_interval=(0.0, 7.0)) -> float:
    """Scales (interpolates) the given number to a given interval.
        :param num: a number
        :param from_interval: the interval the number is assumed to lie in.
        :param to_interval: the target interval.
        :return: the scaled/interpolated number.
    """
    x = np.interp(num, from_interval, to_interval)
    return float(round(x))


def get_blueprint(world: carla.World, id: str) -> carla.ActorBlueprint:
    return world.get_blueprint_library().find(id)


# def save_image(image, name: str, directory='data/images'):
#     def save():
#         cv2.imwrite(os.path.join(directory, name), image, cv2.IMREAD_UNCHANGED)
#         return True
#
#     thread = threading.Thread(target=save)
#     thread.start()

