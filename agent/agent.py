
import tensorflow as tf

from tensorflow.keras.layers import *

from rl import augmentations as aug
from rl.agents import PPOAgent
from rl.agents.imitation import ImitationWrapper

from agent import networks as nn


class CARLAgent(PPOAgent):

    def __init__(self, *args, name='carla-agent', **kwargs):
        super().__init__(*args, name=name, **kwargs)

    def policy_layers(self, inputs: dict, dense_layers=4, dense_units=128, dense_activation='relu', **kwargs) -> Layer:
        print(inputs.keys())
        # Retrieve input layers
        image_in = inputs['state_image']
        road_in = inputs['state_road']
        vehicle_in = inputs['state_vehicle']
        command_in = inputs['state_command']
        control_in = inputs['state_past_control']

        # process features in independence
        image_out = nn.convolutional(image_in, strides=(3, 2), padding='valid', filters=18, units=80, blocks=4,
                                     name='image_out')
        road_out = nn.feature_net(road_in, name='road_out')
        vehicle_out = nn.feature_net(vehicle_in, name='vehicle_out')
        control_out = nn.feature_net(control_in, name='control_out')
        command_out = nn.feature_net(command_in, name='command_out')

        # concatenate outputs
        x = concatenate([image_out, road_out, vehicle_out, control_out, command_out], name='middle_concat')
        x = LayerNormalization()(x)

        # dense layers
        for _ in range(dense_layers):
            x = Dense(dense_units)(x)
            x = ReLU(max_value=6.0, negative_slope=0.1)(x)
            x = Dropout(rate=0.5)(x)

        # use 'command' to implicitly select a "branch" (instead of hand-designing branches like in CIL and CIRL)
        selector = tf.repeat(command_in.input, repeats=32, axis=1)  # 32 = 128 (dense_units) / 4 (command size)
        x = tf.repeat(x, repeats=32 // x.shape[0], axis=0)  # replicating x along batch dimension to match selector

        return multiply([x, selector], name='implicit_branching')


class CARLAImitationLearning(ImitationWrapper):

    def __init__(self, *args, target_size=None, grayscale=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_size = target_size
        self.to_grayscale = grayscale

    def augment(self):
        @tf.function
        def augment_fn(state):
            image = state['state_image']
            image = aug.tf_resize(image, size=self.target_size)

            # contrast, tone, saturation, brightness
            if aug.tf_chance() > 0.5:
                image = aug.tf_saturation(image)

            if aug.tf_chance() > 0.5:
                image = aug.tf_contrast(image, lower=0.5, upper=1.5)

            if aug.tf_chance() > 0.5:
                image = aug.tf_hue(image)

            if aug.tf_chance() > 0.5:
                image = aug.tf_brightness(image, delta=0.5)

            # blur
            if aug.tf_chance() < 0.33:
                image = aug.tf_gaussian_blur(image, size=5)

            # noise
            if aug.tf_chance() < 0.2:
                image = aug.tf_salt_and_pepper(image, amount=0.1)

            if aug.tf_chance() < 0.33:
                image = aug.tf_gaussian_noise(image, amount=0.15, std=0.15)

            image = aug.tf_normalize(image)

            # cutout & dropout
            if aug.tf_chance() < 0.15:
                image = aug.tf_cutout(image, size=6)

            if aug.tf_chance() < 0.10:
                image = aug.tf_coarse_dropout(image, size=49, amount=0.1)

            if self.to_grayscale:
                image = aug.tf_grayscale(image)

            state['state_image'] = 2.0 * image - 1.0  # -1, +1
            return state

        return augment_fn
