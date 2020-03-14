"""Utility functions for environment.py"""

import os
import numpy as np
import tensorflow as tf


def actions_to_control(control, actions):
    pass


def load_nasnet_mobile(folder='weights/keras', weights='nasnet_mobile.h5'):
    path = os.path.join(folder, weights)
    return tf.keras.applications.NASNetMobile(weights=path, pooling='avg')


def get_image_compressor_model(backend='nasnet_mobile', **kwargs):
    if backend == 'nasnet_mobile':
        base_model = load_nasnet_mobile(**kwargs)
    else:
        raise ValueError(f'backend: {backend} is not available!')

    pool2d = base_model.get_layer('global_average_pooling2d')

    return tf.keras.models.Model(inputs=base_model.input,
                                 outputs=pool2d.output,
                                 name=f'{backend}-ImageCompressor')

