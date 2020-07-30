
import tensorflow as tf

from tensorflow.keras.layers import *

from rl import augmentations as aug
from rl.agents import PPOAgent
from rl.agents.imitation import ImitationWrapper

from core import networks as nn


class CARLAgent(PPOAgent):

    def __init__(self, *args, aug_intensity=1.0, name='carla-agent', **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.aug_prob_intensity = aug_intensity

    def policy_layers(self, inputs: dict, dense_layers=4, dense_units=98, **kwargs) -> Layer:
        # Retrieve input layers
        image_in = inputs['state_image']
        road_in = inputs['state_road']
        vehicle_in = inputs['state_vehicle']
        command_in = inputs['state_command']
        control_in = inputs['state_past_control']

        # process features in independence
        image_out = nn.convolutional(image_in, strides=(2, 3), padding='valid', filters=14, units=96, blocks=4,
                                     name='image_out')
        road_out = nn.feature_net(road_in, units=32, dropout=0.4, name='road_out')
        vehicle_out = nn.feature_net(vehicle_in, units=24, dropout=0.4, name='vehicle_out')
        control_out = nn.feature_net(control_in, units=16, dropout=0.4, name='control_out')
        command_out = nn.feature_net(command_in, units=16, dropout=0.4, name='command_out')

        # concatenate outputs
        x = concatenate([image_out, road_out, vehicle_out, control_out, command_out], name='middle_concat')
        x = LayerNormalization()(x)

        # dense layers
        for _ in range(dense_layers):
            x = Dense(dense_units)(x)
            x = ReLU(max_value=6.0, negative_slope=0.1)(x)
            x = Dropout(rate=0.4)(x)

        # use 'command' to implicitly select a "branch" (instead of hand-designing branches like in CIL and CIRL)
        repetitions = dense_units // command_in.shape[-1]
        selector = tf.repeat(command_in, repeats=repetitions, axis=1)
        x = tf.repeat(x, repeats=self.batch_size // x.shape[0], axis=0)  # repeat 'x' along batch dimension

        return multiply([selector, x], name='implicit_branching')

    def preprocess(self):
        alpha = self.aug_prob_intensity

        @tf.function
        def augment_fn(state, _):
            state = state.copy()
            image = state['state_image']
            image = tf.image.grayscale_to_rgb(image)

            # contrast, tone, saturation, brightness
            if aug.tf_chance() > 0.5 * alpha:
                image = aug.tf_saturation(image)

            if aug.tf_chance() > 0.5 * alpha:
                image = aug.tf_contrast(image, lower=0.5, upper=1.5)

            if aug.tf_chance() > 0.5 * alpha:
                image = aug.tf_hue(image)

            if aug.tf_chance() > 0.5 * alpha:
                image = aug.tf_brightness(image, delta=0.5)

            # blur
            if aug.tf_chance() < 0.33 * alpha:
                image = aug.tf_gaussian_blur(image, size=5)

            # noise
            if aug.tf_chance() < 0.2 * alpha:
                image = aug.tf_salt_and_pepper(image, amount=0.1)

            if aug.tf_chance() < 0.33 * alpha:
                image = aug.tf_gaussian_noise(image, amount=0.15, std=0.15)

            image = aug.tf_normalize(image)

            # cutout & dropout
            if aug.tf_chance() < 0.15 * alpha:
                image = aug.tf_cutout(image, size=6)

            if aug.tf_chance() < 0.10 * alpha:
                image = aug.tf_coarse_dropout(image, size=49, amount=0.1)

            image = 2.0 * aug.tf_grayscale(image) - 1.0  # -1, +1
            state['state_image'] = image
            return state, _


class CARLAImitationLearning(ImitationWrapper):

    def __init__(self, *args, target_size=None, aug_intensity=1.0, grayscale=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_size = target_size
        self.to_grayscale = grayscale
        self.aug_prob_intensity = aug_intensity

    def preprocess(self):
        # target_size = tf.TensorShape(dims=self.target_size)
        should_grayscale = tf.constant(self.to_grayscale, dtype=tf.bool)
        alpha = self.aug_prob_intensity

        @tf.function
        def augment_fn(state, _):
            state = state.copy()
            image = state['state_image']
            # image = aug.tf_resize(image, size=target_size)
            image = tf.image.grayscale_to_rgb(image)

            # contrast, tone, saturation, brightness
            if aug.tf_chance() > 0.5 * alpha:
                image = aug.tf_saturation(image)

            if aug.tf_chance() > 0.5 * alpha:
                image = aug.tf_contrast(image, lower=0.5, upper=1.5)

            if aug.tf_chance() > 0.5 * alpha:
                image = aug.tf_hue(image)

            if aug.tf_chance() > 0.5 * alpha:
                image = aug.tf_brightness(image, delta=0.5)

            # blur
            if aug.tf_chance() < 0.33 * alpha:
                image = aug.tf_gaussian_blur(image, size=5)

            # noise
            if aug.tf_chance() < 0.2 * alpha:
                image = aug.tf_salt_and_pepper(image, amount=0.1)

            if aug.tf_chance() < 0.33 * alpha:
                image = aug.tf_gaussian_noise(image, amount=0.15, std=0.15)

            image = aug.tf_normalize(image)

            # cutout & dropout
            if aug.tf_chance() < 0.15 * alpha:
                image = aug.tf_cutout(image, size=6)

            if aug.tf_chance() < 0.10 * alpha:
                image = aug.tf_coarse_dropout(image, size=49, amount=0.1)

            if should_grayscale:
                image = aug.tf_grayscale(image)

            state['state_image'] = 2.0 * image - 1.0  # -1, +1
            return state, _

        return augment_fn
