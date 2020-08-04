import os
import gym
import time
import tensorflow as tf
import numpy as np

from typing import Union

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

from rl import utils
from rl import augmentations as aug
from rl.agents import PPOAgent
from rl.environments import ThreeCameraCARLAEnvironmentDiscrete
from rl.parameters.schedules import Schedule

from core.networks import CARLANetwork


class FakeCARLAEnvironment(gym.Env):
    """A testing-only environment with the same state- and action-space of a CARLA Environment"""

    def __init__(self):
        super().__init__()
        env = ThreeCameraCARLAEnvironmentDiscrete

        self.action_space = env.ACTION['space']
        self.observation_space = gym.spaces.Dict(road=env.ROAD_FEATURES['space'],
                                                 vehicle=env.VEHICLE_FEATURES['space'],
                                                 past_control=env.CONTROL['space'], command=env.COMMAND_SPACE,
                                                 image=gym.spaces.Box(low=-1.0, high=1.0, shape=(90, 360, 3)))

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass


class CARLAgent(PPOAgent):
    # predefined architecture:
    DEFAULT_CONTROL = dict(num_layers=4 - 2, units_multiplier=16)
    DEFAULT_DYNAMICS = dict(road=dict(num_layers=3, units=16),
                            vehicle=dict(num_layers=2, units=16),
                            command=dict(num_layers=2, units=16),
                            shufflenet=dict(linear_units=128, g=0.5, last_channels=768),
                            value=dict(linear_units=0, units=8),
                            action=dict(linear_units=32, units=16))

    def __init__(self, *args, tau=0.1, context_size=64, aug_intensity=1.0, clip_norm=(1.0, 1.0, 1.0), name='carla-ppo',
                 dynamics_lr: Union[float, LearningRateSchedule] = 1e-3, **kwargs):
        assert aug_intensity >= 0.0

        # network specification
        network_spec = kwargs.get('network', {})
        network_spec.setdefault('network', CARLANetwork)
        network_spec.setdefault('context_size', context_size)
        network_spec.setdefault('control', self.DEFAULT_CONTROL)
        network_spec.setdefault('dynamics', self.DEFAULT_DYNAMICS)

        super().__init__(*args, name=name, network=network_spec, clip_norm=clip_norm, **kwargs)

        self.weights_path['dynamics'] = os.path.join(self.base_path, 'dynamics_model')
        self.network: CARLANetwork = self.network

        # self.aug_intensity = tf.constant(aug_intensity, dtype=tf.float32)
        self.aug_intensity = aug_intensity
        self.tau = tf.constant(tau, dtype=tf.float32)

        # gradient clipping:
        if (clip_norm is None) or (len(clip_norm) < 3) or (clip_norm[2] is None):
            self.should_clip_dynamics_grads = False
        else:
            assert isinstance(clip_norm[2], float)

            self.should_clip_dynamics_grads = True
            self.grad_norm_dynamics = tf.constant(clip_norm[2], dtype=tf.float32)

        # optimizer
        self.dynamics_optimizer = utils.get_optimizer_by_name(name=kwargs.get('optimizer', 'adam'),
                                                              learning_rate=dynamics_lr)
        self.dynamics_lr = dynamics_lr
        self.has_schedule_dynamics = isinstance(dynamics_lr, Schedule)

    def predict(self, state):
        result = super().predict(state)
        self.log(context=self.network.get_context())
        return result

    def act(self, state: dict):
        raise NotImplementedError

    def policy_batch_tensors(self, advantages) -> Union[tuple, dict]:
        values = self.memory.values[1:]
        return self.memory.states, advantages, self.memory.actions, self.memory.log_probabilities, values

    def get_value_batches(self, returns, **kwargs):
        """Computes batches of data for updating the value network"""
        return utils.data_to_batches(tensors=returns, batch_size=self.batch_size,
                                     drop_remainder=self.drop_batch_reminder, skip=self.skip_count,
                                     num_shards=self.obs_skipping, shuffle_batches=self.shuffle_batches)

    def learn_representation(self, num_traces=np.inf, batch_size=None, shuffle_batches=False, shuffle_traces=False,
                             seed=None, save_every: Union[str, None] = 'end', **kwargs):
        """Implements SimCLR-like training (see https://arxiv.org/pdf/2002.05709) for representation learning"""
        print('== REPRESENTATION LEARNING ==')

        self.set_random_seed(seed)

        if self.network.projection is None:
            self.network.projection_model(**kwargs)
        else:
            self.network.reset_projection()

        for i, trace in enumerate(utils.load_traces(self.traces_dir, num_traces, shuffle=shuffle_traces)):
            t0 = time.time()

            # load trace:
            states, actions, rewards, _ = utils.unpack_trace(trace)

            returns = utils.rewards_to_go(rewards, discount=self.gamma, normalize=False)
            self.returns.update(returns)
            returns = self.returns.normalize(returns)

            states['action'] = actions
            states['value'] = returns

            # bootstrap mean and std of advantages (for later RL)
            adv = utils.gae(rewards, returns, gamma=self.gamma, lambda_=self.lambda_, normalize=False)
            self.advantages.update(adv)
            adv = self.advantages.normalize(adv)

            self.log(actions=actions, returns=states['value'], advantages=adv)

            # optimization:
            self.simclr_update(data=states, batch_size=self.batch_size if batch_size is None else batch_size,
                               shuffle_batches=shuffle_batches, seed=seed)
            # # saving:
            # if (i + 1) % save_every == 0:
            #     self.save()

            self.write_summaries()

            print(f'Episode took {round(time.time() - t0, 3)}s.')

        if save_every == 'end':
            self.save()

    def simclr_update(self, data, batch_size: int, shuffle_batches=False, seed=None):
        t0 = time.time()
        batches = utils.data_to_batches(data, batch_size, shuffle_batches=shuffle_batches, seed=seed)

        for batch in batches:
            batch1 = tf.map_fn(fn=self.augment(), elems=batch)
            batch2 = tf.map_fn(fn=self.augment(), elems=batch)

            with tf.GradientTape() as tape:
                predictions = tf.concat([
                    self.network.projection_predict(batch1),
                    self.network.projection_predict(batch2)], axis=1)

                loss = self.simclr_objective(predictions)

            gradients = tape.gradient(loss, self.network.projection.trainable_variables)

            if self.should_clip_dynamics_grads:
                gradients = [tf.clip_by_norm(grad, clip_norm=self.grad_norm_dynamics) for grad in gradients]

            self.dynamics_optimizer.apply_gradients(zip(gradients, self.network.projection.trainable_variables))

            self.log(loss_dynamics=loss, gradients_norm_dynamics=[tf.norm(gradient) for gradient in gradients],
                     lr_dynamics=self.dynamics_lr.lr if self.has_schedule_dynamics else self.dynamics_lr,
                     predictions_norm=[tf.norm(p) for p in predictions])

        print(f'Dynamics update took {round(time.time() - t0, 3)}s.')

    def simclr_objective(self, batch):
        def pairwise_cosine_similarities(data, eps=utils.EPSILON):
            # pairwise dot-product as matrix multiplication by its transpose:
            pairwise_dot_prod = tf.matmul(data, data, transpose_b=True)

            # pairwise vector-norms:
            norms = tf.norm(data, axis=1)  # for each vector/row (axis=1) compute its norm
            pairwise_norms = norms * tf.reshape(norms, shape=(norms.shape[0], 1))  # norm * transpose(norm)

            # pairwise similarities:
            pairwise_sims = pairwise_dot_prod / (pairwise_norms + eps)
            return tf.clip_by_value(pairwise_sims, -1.0, 1.0)

        def normalized_temperature_scaled_cross_entropy_loss(i, j):
            numerator = similarities[i, j] / self.tau
            denominator = tf.reduce_sum(exp_sim_tau[i])

            return -numerator + tf.math.log(denominator - exp_sim_tau[i, i])

        batch_size = batch.shape[0]
        similarities = pairwise_cosine_similarities(batch)
        exp_sim_tau = tf.exp(similarities / self.tau)
        loss = 0.0

        # for i in range(batch_size):
        #     for j in range(batch_size):
        #         self.log(similarity=similarities[i, j])

        # compute losses:
        half_batch = batch_size // 2
        for k in range(half_batch):
            i = k
            j = k + half_batch

            l1 = normalized_temperature_scaled_cross_entropy_loss(i, j)
            l2 = normalized_temperature_scaled_cross_entropy_loss(j, i)
            loss += l1 + l2

            # self.log(loss_1=l1, loss_2=l2)

        return loss / batch_size

    def augment(self):
        alpha = self.aug_intensity

        @tf.function
        def augment_fn(state):
            state = state.copy()
            image = state['state_image']

            # TODO: add random seed
            size = aug.tf_scale_shape(image, scale=(0.5, 0.33))
            image = aug.simclr.pipeline(image, crop_size=size, strength=alpha, blur_size=0.02, blur_sigma=(0.1, 1.0))

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

            state['state_image'] = 2.0 * image - 1.0  # -1, +1
            return state

        return augment_fn

    def preprocess(self):
        alpha = self.aug_intensity

        @tf.function
        def augment_fn(state, *args):
            state = state.copy()
            image = state['state_image']

            # Color distortion
            image = aug.simclr.color_distortion(image, strength=alpha)

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

            state['state_image'] = 2.0 * image - 1.0  # -1, +1
            return (state, *args)

        return augment_fn


if __name__ == '__main__':
    # agent = CARLAgent(FakeCARLAEnvironment(), batch_size=32, log_mode=None)
    # agent.summary()

    agent = CARLAgent(ThreeCameraCARLAEnvironmentDiscrete(bins=6, image_shape=(90, 120, 3), window_size=(720, 180)),
                      batch_size=32 * 2, log_mode='summary', seed=123, optimization_steps=(1, 1))
    # agent.summary()
    # breakpoint()
    agent.learn(episodes=5, timesteps=512 // 2)
    pass
