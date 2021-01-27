
import tensorflow as tf

from rl.augmentations.augmentations import *


# -------------------------------------------------------------------------------------------------
# -- SimCLR Augmentations (see https://arxiv.org/pdf/2002.05709, appendix A)
# -------------------------------------------------------------------------------------------------


def pipeline(image, crop_size: Size, strength=1.0, flip_prob=0.5, jitter_prob=0.8, drop_prob=0.2, blur_prob=0.5,
             blur_size=0.1, blur_sigma=(0.1, 2.0), seed=None):
    """Applies the following SimCLR-like augmentations:
         - crop_resize_flip -> color_distortion -> gaussian_blur
    """
    image = crop_resize_flip(image, size=crop_size, flip_prob=flip_prob, seed=seed)
    image = color_distortion(image, strength=strength, jitter_prob=jitter_prob, drop_prob=drop_prob, seed=seed)
    image = gaussian_blur(image, size=blur_size, sigma=blur_sigma, prob=blur_prob, seed=seed)
    return image


def crop_resize_flip(image, size: Size, flip_prob=0.5, seed=None):
    """Crops the image, then resize it back to the original shape, finally applies horizontal flipping."""
    image = tf_crop(image, size, resize=True, seed=seed)

    if tf_chance(seed) <= flip_prob:
        return tf_flip(image, horizontal=True)

    return image


def color_distortion(image, strength=1.0, jitter_prob=0.8, drop_prob=0.2, seed=None):
    """Color distortion (jitter + drop) augmentation as defined in SimCLR paper"""
    if tf_chance(seed) <= jitter_prob:
        image = color_jitter(image, strength, seed)

    if tf_chance(seed) <= drop_prob:
        image = color_drop(image)

    return image


@tf.function
def color_jitter(image, strength=1.0, original=True, seed=None):
    """Color jitter augmentation as defined in SimCLR paper"""
    image = tf_brightness(image, delta=0.2 * strength, seed=seed)
    image = tf_contrast(image, lower=1.0 - 0.8 * strength, upper=1.0 + 0.8 * strength, seed=seed)
    image = tf_saturation(image, lower=1.0 - 0.8 * strength, upper=1.0 + 0.8 * strength, seed=seed)
    image = tf_hue(image, delta=0.2 * strength, seed=seed)

    if original:
        return tf.clip_by_value(image, 0.0, 1.0)

    if len(image.shape) == 4:
        return tf_normalize_batch(image)

    return tf_normalize(image)


@tf.function
def color_drop(image):
    """Color drop augmentation as defined in SimCLR paper"""
    return tf_repeat_channels(tf_grayscale(image), n=3)


def gaussian_blur(image, size=0.1, sigma=(0.1, 2.0), prob=0.5, seed=None):
    """Gaussian blur augmentation as defined in SimCLR paper"""
    if tf_chance(seed) <= prob:
        kernel_size = tf.cast(tf.minimum(image.shape[0] * size, image.shape[1] * size), dtype=tf.int32)
        std = tf.random.shuffle(sigma, seed=seed)[0]

        return tf_gaussian_blur(image, size=kernel_size, std=std, seed=seed)

    return image
