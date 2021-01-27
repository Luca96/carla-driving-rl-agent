"""Data augmentations based on tf's functions"""

import tensorflow as tf

from typing import Union, List, Tuple

from rl import utils


Size = Union[List[int], Tuple[int, ...], tf.TensorShape]


# -------------------------------------------------------------------------------------------------
# -- Geometric/Spatial Augmentations
# -------------------------------------------------------------------------------------------------

def tf_resize(image, size: Size):
    return tf.image.resize(image, size)


def tf_crop(image, size: Size, resize=False, seed=None):
    cropped = tf.image.random_crop(image, size, seed=seed)

    if resize:
        return tf_resize(cropped, size=image.shape[:2])

    return cropped


def tf_flip(image, horizontal=True, vertical=False, seed=None):
    if horizontal:
        image = tf.image.random_flip_left_right(image, seed=seed)

    if vertical:
        image = tf.image.random_flip_up_down(image, seed=seed)

    return image


def tf_quality(image, min_quality: int, max_quality: int, seed=None):
    return tf.image.random_jpeg_quality(image, min_jpeg_quality=min_quality, max_jpeg_quality=max_quality, seed=seed)


@tf.function
def tf_cutout(image, size=5, seed=None):
    cut_mask = tf.random.normal(shape=(size, size), seed=seed)
    cut_mask = tf.where(condition=cut_mask == tf.reduce_max(cut_mask), x=0.0, y=1.0)
    cut_mask = tf.stack((cut_mask,) * 3, axis=-1)
    cut_mask = tf.image.resize([cut_mask], size=image.shape[:2],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[0]
    return image * cut_mask


@tf.function
def tf_cutout_batch(images, size=5, seed=None):
    masks = []

    for _ in range(images.shape[0]):
        cut_mask = tf.random.normal(shape=(size, size), seed=seed)
        cut_mask = tf.where(condition=cut_mask == tf.reduce_max(cut_mask), x=0.0, y=1.0)
        cut_mask = tf.stack((cut_mask,) * 3, axis=-1)
        masks.append(cut_mask)

    masks = tf.stack(masks, axis=0)
    masks = tf.image.resize(masks, size=images.shape[1:3],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[0]
    return images * masks


@tf.function
def tf_coarse_dropout(image, size=25, amount=0.1, seed=None):
    drop_mask = tf.keras.backend.random_binomial((size, size), p=1.0 - amount, seed=seed)
    drop_mask = tf.stack((drop_mask,) * 3, axis=-1)
    drop_mask = tf.image.resize([drop_mask], size=image.shape[:2],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[0]
    return image * drop_mask


@tf.function
def tf_coarse_dropout_batch(images, size=25, amount=0.1, seed=None):
    masks = []

    for _ in range(images.shape[0]):
        drop_mask = tf.keras.backend.random_binomial((size, size), p=1.0 - amount, seed=seed)
        drop_mask = tf.stack((drop_mask,) * 3, axis=-1)
        masks.append(drop_mask)

    masks = tf.stack(masks, axis=0)
    masks = tf.image.resize(masks, size=images.shape[1:3],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[0]
    return images * masks


def tf_rotate(image, degrees=90):
    assert degrees % 90 == 0
    return tf.image.rot90(image, k=degrees // 90)


# -------------------------------------------------------------------------------------------------
# -- Appearance Augmentations
# -------------------------------------------------------------------------------------------------

def tf_saturation(image, lower=0.5, upper=1.5, seed=None):
    return tf.image.random_saturation(image, lower, upper, seed=seed)


def tf_contrast(image, lower=0.4, upper=1.6, seed=None):
    return tf.image.random_contrast(image, lower, upper, seed=seed)


def tf_brightness(image, delta=0.75, seed=None):
    return tf.image.random_brightness(image, max_delta=delta, seed=seed)


def tf_hue(image, delta=0.5, seed=None):
    return tf.image.random_hue(image, max_delta=delta, seed=seed)


def tf_grayscale(rgb_image):
    return tf.image.rgb_to_grayscale(rgb_image)


def tf_rgb(gray_image):
    return tf.image.grayscale_to_rgb(gray_image)


@tf.function
def tf_gaussian_noise(image, amount=0.25, std=0.2, seed=None):
    mask_select = tf.keras.backend.random_binomial(image.shape[:2], p=amount, seed=seed)
    mask_select = tf.stack((mask_select,) * 3, axis=-1)

    mask_noise = tf.random.normal(shape=image.shape, stddev=std, seed=seed)
    mask_noise = tf.clip_by_value(mask_noise, 0.0, 1.0)

    return image + (mask_select * mask_noise)


@tf.function
def tf_gaussian_noise_batch(images, amount=0.25, std=0.2, seed=None):
    masks = []

    for _ in range(images.shape[0]):
        mask_select = tf.keras.backend.random_binomial(images.shape[1:3], p=amount, seed=seed)
        mask_select = tf.stack((mask_select,) * 3, axis=-1)
        mask_noise = tf.random.normal(shape=images.shape[1:], stddev=std, seed=seed)

        masks.append(tf.clip_by_value(mask_select * mask_noise, 0.0, 1.0))

    return images + tf.stack(masks, axis=0)


@tf.function
def tf_salt_and_pepper(image, amount=0.1, prob=0.5, seed=None):
    # source: https://stackoverflow.com/questions/55653940/how-do-i-implement-salt-pepper-layer-in-keras
    mask_select = tf.keras.backend.random_binomial(image.shape[:2], p=amount / 10, seed=seed)
    mask_select = tf.stack((mask_select,) * 3, axis=-1)

    mask_noise = tf.keras.backend.random_binomial(image.shape[:2], p=prob, seed=seed)
    mask_noise = tf.stack((mask_noise,) * 3, axis=-1)
    return image * (1 - mask_select) + mask_noise * mask_select


@tf.function
def tf_salt_and_pepper_batch(images, amount=0.1, prob=0.5, seed=None):
    # source: https://stackoverflow.com/questions/55653940/how-do-i-implement-salt-pepper-layer-in-keras
    masks_select = []
    masks_noise = []

    for _ in range(images.shape[0]):
        mask_select = tf.keras.backend.random_binomial(images.shape[1:3], p=amount / 10, seed=seed)
        mask_select = tf.stack((mask_select,) * 3, axis=-1)
        masks_select.append(mask_select)

        mask_noise = tf.keras.backend.random_binomial(images.shape[1:3], p=prob, seed=seed)
        mask_noise = tf.stack((mask_noise,) * 3, axis=-1)
        masks_noise.append(mask_noise)

    mask_select = tf.stack(masks_select, axis=0)
    mask_noise = tf.stack(masks_noise, axis=0)

    return images * (1 - mask_select) + mask_noise * mask_select


@tf.function
def tf_gaussian_blur(image, size=5, std=0.25, seed=None):
    # source: https://gist.github.com/blzq/c87d42f45a8c5a53f5b393e27b1f5319
    gaussian_kernel = tf.random.normal(shape=(size, size, image.shape[-1], 1), mean=1.0, stddev=std, seed=seed)

    if len(image.shape) == 3:
        image = tf.expand_dims(image, axis=0)
        image = tf.nn.depthwise_conv2d(image, gaussian_kernel, [1, 1, 1, 1], padding='SAME',
                                       data_format='NHWC')[0]
    else:
        image = tf.nn.depthwise_conv2d(image, gaussian_kernel, [1, 1, 1, 1], padding='SAME',
                                       data_format='NHWC')
    return image


@tf.function
def tf_median_blur(image, size=5):
    median_kernel = tf.ones((size, size, image.shape[-1], 1))

    if len(image.shape) == 3:
        image = tf.expand_dims(image, axis=0)
        image = tf.nn.depthwise_conv2d(image, median_kernel, [1, 1, 1, 1], padding='SAME',
                                       data_format='NHWC')[0]
    else:
        image = tf.nn.depthwise_conv2d(image, median_kernel, [1, 1, 1, 1], padding='SAME',
                                       data_format='NHWC')
    return image


@tf.function
def tf_multiply_channels(image, strength=1.0, seed=None):
    """Channel-wise multiplication of given image by random scalars. The scalars sum to one, each scalar multiplies
       an entire channel
    """
    assert len(image.shape) == 3

    logits = tf.random.uniform(shape=(image.shape[2],), minval=-1, maxval=1, seed=seed)
    alpha = tf.nn.softmax(logits) * strength

    return tf_normalize(image * alpha)


@tf.function
def tf_sobel(image, grayscale=False, restore_depth=True, normalize=True):
    """Applies Sobel filtering"""
    if grayscale:
        depth = image.shape[2]
        image = tf_grayscale(image)

    image = tf.image.sobel_edges(tf.expand_dims(image, axis=0))
    dx, dy = tf.unstack(image[0], axis=-1)
    result = dx + dy

    if grayscale and restore_depth:
        result = tf_repeat_channels(result, n=depth)

    if normalize:
        return tf_normalize(result)

    return result


# -------------------------------------------------------------------------------------------------

@tf.function
def tf_normalize(image, eps=utils.EPSILON):
    """Scales the given image in range [0.0, 1.0]"""
    image -= tf.reduce_min(image)
    image /= tf.reduce_max(image) + eps
    return image


@tf.function
def tf_normalize_batch(images):
    return tf.map_fn(fn=tf_normalize, elems=images)


def tf_chance(seed=None):
    """Use to get a single random number between 0 and 1"""
    return tf.random.uniform(shape=(1,), minval=0.0, maxval=1.0, seed=seed)


@tf.function
def tf_repeat_channels(image, n=3):
    if len(image.shape) == 2:
        return tf.stack((image,) * n, axis=-1)

    return tf.concat((image,) * n, axis=-1)


def tf_scale_shape(image, scale: Tuple[float, float]):
    h, w, d = image.shape
    return utils.to_int((h * scale[0], w * scale[0], d))


def tf_size(image):
    return image.shape[:2]
