import os
import gym
import time
import random
import tensorflow as tf
import numpy as np
import json
import rl

from typing import Union, List

from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras import losses

from rl import utils, augmentations as aug
from rl import PPOAgent
from rl import PPOMemory
from rl import ThreeCameraCARLAEnvironmentDiscrete
from rl.environments.carla.tools import utils as carla_utils
from rl.parameters import DynamicParameter
from gym import spaces

from core.networks import CARLANetwork


class FakeCARLAEnvironment(gym.Env):
    """A testing-only environment with the same state- and action-space of a CARLA Environment"""

    def __init__(self):
        super().__init__()
        env = ThreeCameraCARLAEnvironmentDiscrete
        self.num_waypoints = 10
        self.NAVIGATION_FEATURES = {}
        self.NAVIGATION_FEATURES['space'] = spaces.Box(low=0.0, high=25.0, shape=(self.num_waypoints,))
        self.NAVIGATION_FEATURES['default'] = np.zeros(shape=self.num_waypoints, dtype=np.float32)

        self.time_horizon = 1
        self.action_space = env.ACTION['space']
        self.observation_space = gym.spaces.Dict(road=env.ROAD_FEATURES['space'],
                                                 vehicle=env.VEHICLE_FEATURES['space'],
                                                 past_control=env.CONTROL['space'], command=env.COMMAND_SPACE,
                                                 image=gym.spaces.Box(low=-1.0, high=1.0, shape=(90, 360, 3)),
                                                 navigation=self.NAVIGATION_FEATURES['space'])

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass


# -------------------------------------------------------------------------------------------------
# -- Agent
# -------------------------------------------------------------------------------------------------

class CARLAgent(PPOAgent):
    # Default neural network architecture
    DEFAULT_CONTROL = dict(units=320, num_layers=2, activation=utils.swish6)
    DEFAULT_CONTROL_VALUE = dict(units=320, num_layers=2, activation=utils.swish6)
    DEFAULT_DYNAMICS = dict(road=dict(units=16, num_layers=2, activation=tf.nn.relu6),
                            vehicle=dict(units=16, num_layers=2, activation=tf.nn.relu6),
                            navigation=dict(units=16, num_layers=2, activation=tf.nn.relu6),
                            shufflenet=dict(g=1.0, last_channels=768),
                            rnn=dict(image=256, road=32, vehicle=32, navigation=32),
                            dynamics=dict(units=512))

    def __init__(self, *args, aug_intensity=1.0, clip_norm=(1.0, 1.0, 1.0), name='carla', load_full=True, eta=0.0,
                 dynamics_lr: Union[float, LearningRateSchedule] = 1e-3, update_dynamics=True, delta=0.0, aux=1.0,
                 **kwargs):
        """
        :param aug_intensity: how much intense the augmentation should be.
        :param clip_norm: at which value the gradient's norm should be clipped.
        :param name: the agent's name, used for saving and loading.
        :param load_full: whether or not to load the full network (dynamics + value + policy) or just the dynamics net.
        :param eta: [experimental] scalar that encourages the agent to accelerate (see `policy_objective`)
        :param dynamics_lr: learning rate of the shared dynamics network.
        :param update_dynamics: whether or not to also update the shared network during a reinforcement learning update.
        :param delta: [experimental] scalar that penalizes the steering magnitude (see `policy_objective`)
        :param aux: use to balance auxiliary losses (speed + similarity loss).
        """
        assert aug_intensity >= 0.0

        # network specification
        network_spec = kwargs.pop('network', {})
        network_spec.setdefault('network', CARLANetwork)
        network_spec.setdefault('control_policy', self.DEFAULT_CONTROL)
        network_spec.setdefault('control_value', self.DEFAULT_CONTROL_VALUE)
        network_spec.setdefault('dynamics', self.DEFAULT_DYNAMICS)

        self.should_update_dynamics = update_dynamics
        self.dynamics_path = os.path.join(kwargs.get('weights_dir', 'weights'), name, 'dynamics_model')
        self.load_full = load_full

        super().__init__(*args, name=name, network=network_spec, clip_norm=clip_norm, **kwargs)

        self.network: CARLANetwork = self.network
        self.aug_intensity = aug_intensity

        self.delta = delta
        self.eta = eta
        self.aux = aux

        self.evaluation_path = utils.makedir(os.path.join(self.base_path, 'evaluation'))

        # gradient clipping:
        if isinstance(clip_norm, float):
            self.should_clip_dynamics_grads = True
            self.grad_norm_dynamics = tf.constant(clip_norm, dtype=tf.float32)

        elif isinstance(clip_norm[2], float):
            self.should_clip_dynamics_grads = True
            self.grad_norm_dynamics = tf.constant(clip_norm[2], dtype=tf.float32)
        else:
            self.should_clip_dynamics_grads = False

        # optimizer
        self.dynamics_lr = DynamicParameter.create(value=dynamics_lr)
        self.dynamics_lr.load(config=self.config.get('dynamics_lr', {}))

        self.dynamics_optimizer = utils.get_optimizer_by_name(name=kwargs.get('optimizer', 'adam'),
                                                              learning_rate=dynamics_lr)

        self.network.policy.compile(optimizer=self.policy_optimizer, run_eagerly=False)
        self.network.old_policy.compile(optimizer=self.policy_optimizer, run_eagerly=False)

    def update(self):
        if len(self.memory) < self.batch_size:
            print('[Not updated] memory too small!')
            self.env.reset_info()
            return

        super().update()

        try:
            actions = self.memory.actions
            actions = (actions - 1.0) * 2.0 + 1.0
            self.log(action_throttle_or_brake=actions[:, 0], action_steer=actions[:, 1])

        except Exception:
            print('[update] unable to print actions')

        self.env.reset_info()

    def record(self, name: str, timesteps: int, trials: int, seed=None, sample_seed=True, close=False):
        assert trials > 0
        assert timesteps > 0

        if isinstance(seed, int):
            self.set_random_seed(seed=seed)

        try:
            for trial in range(1, trials + 1):
                self.memory = self.get_memory()

                if sample_seed:
                    self.set_random_seed(seed=random.randint(a=0, b=2**32 - 1))

                record_path = utils.makedir(os.path.join('record', self.env.current_town, name, str(trial)))
                self.env.set_record_path(path=record_path) 

                preprocess_fn = self.preprocess()
                self.reset()

                state = self.env.reset()
                t0 = time.time()
                total_reward = 0.0

                for t in range(1, timesteps + 1):
                    if isinstance(state, dict):
                        state = {f'state_{k}': v for k, v in state.items()}

                    state = preprocess_fn(state)
                    state = utils.to_tensor(state)

                    # Agent prediction
                    action, mean, std, log_prob, value = self.predict(state)
                    action_env = self.convert_action(action)

                    # Environment step
                    next_state, reward, done, _ = self.env.step(action_env)
                    total_reward += reward

                    self.memory.append(state, action, reward, value, log_prob)
                    state = next_state

                    # use t > 1 to skip accidental collision at first timestep
                    if (done or (t == timesteps)) and (t > 32):
                        print(f'Trial-{trial} terminated after {t} timesteps in {round(time.time() - t0, 3)} '
                              f'with total reward of {round(total_reward, 3)}.')

                        with open(os.path.join(record_path, 'info.json'), 'w') as f:
                            data = dict(reward=total_reward, timestep=t, weather=str(self.env.weather))
                            json.dump(data, fp=f, indent=3)

                        break

                self.memory.delete()
        finally:
            if close:
                self.env.close()

    def evaluate(self, name: str, timesteps: int, trials: int, seeds: Union[None, List[int]] = None,
                 town='Town03', initial_seed=None, close=False) -> dict:
        assert trials > 0
        assert timesteps > 0

        if isinstance(initial_seed, int):
            self.set_random_seed(seed=initial_seed)

        if town is not None:
            self.env.set_town(town)
        # self.env.set_weather(weather)

        results = dict(collision_rate=[], similarity=[], waypoint_distance=[],
                       speed=[], total_reward=[], timesteps=[])
        save_path = os.path.join(self.evaluation_path, f'{name}.json')
        print(save_path)

        # disable data-augmentation
        aug_intensity = self.aug_intensity
        self.aug_intensity = 0.0

        try:
            trials_done = 0

            while trials_done < trials:
                self.memory = self.get_memory()

                # random seed
                if isinstance(seeds, list):
                    if len(seeds) == trials:
                        self.set_random_seed(seed=seeds[trials_done])
                    else:
                        self.set_random_seed(seed=random.choice(seeds))

                elif seeds == 'sample':
                    self.set_random_seed(seed=random.randint(a=0, b=2 ** 32 - 1))

                preprocess_fn = self.preprocess()
                self.reset()

                state = self.env.reset()
                t0 = time.time()
                total_reward = 0.0

                # vehicle statistics
                similarity = 0.0
                speed = 0.0
                waypoint_distance = 0.0

                for t in range(1, timesteps + 1):
                    if isinstance(state, dict):
                        state = {f'state_{k}': v for k, v in state.items()}

                    state = preprocess_fn(state)
                    state = utils.to_tensor(state)

                    # Agent prediction
                    action, mean, std, log_prob, value = self.predict(state)
                    action_env = self.convert_action(action)

                    # Environment step
                    for _ in range(self.repeat_action):
                        next_state, reward, done, _ = self.env.step(action_env)
                        total_reward += reward

                        if done:
                            break

                    similarity += self.env.similarity
                    speed += carla_utils.speed(self.env.vehicle)
                    waypoint_distance += self.env.route.distance_to_next_waypoint()

                    self.log(eval_actions=action, eval_rewards=reward,
                             eval_distribution_mean=mean, eval_distribution_std=std)

                    self.memory.append(state, action, reward, value, log_prob)
                    state = next_state

                    # use t > 1 to skip accidental collision at first timestep
                    if (done or (t == timesteps)) and (t > 32):
                        # save results of current trial
                        results['total_reward'].append(-1000.0 if total_reward < -1000.0 else total_reward)
                        results['timesteps'].append(t)
                        results['collision_rate'].append(1.0 if self.env.should_terminate else 0.0)
                        results['similarity'].append(similarity / t)
                        results['waypoint_distance'].append(waypoint_distance / t)
                        results['speed'].append(speed / t)

                        self.log(**{f'eval_{k}': v[-1] for k, v in results.items()})

                        print(f'Trial-{trials_done} terminated after {t} timesteps in {round(time.time() - t0, 3)} '
                              f'with total reward of {round(total_reward, 3)}.')
                        trials_done += 1
                        break

                self.write_summaries()
                self.memory.delete()

            # save average with standard-deviation of results over trials as json
            record = dict()

            for k, v in results.items():
                record[k] = v
                record[f'{k}_mean'] = float(np.mean(v))
                record[f'{k}_std'] = float(np.std(v))

            with open(save_path, 'w') as file:
                json.dump(record, fp=file, indent=2)

        finally:
            # restore data-augmentation intensity value
            self.aug_intensity = aug_intensity

            if close:
                self.env.close()

        return results

    def policy_batch_tensors(self) -> Union[tuple, dict]:
        """Defines a batch of data for the policy network"""
        states, advantages, actions, log_probabilities = super().policy_batch_tensors()
        states['action'] = actions

        speed = utils.to_tensor(self.env.info_buffer['speed'], expand_axis=-1) / 100.0
        similarity = utils.to_tensor(self.env.info_buffer['similarity'], expand_axis=-1)

        return states, advantages, log_probabilities, speed, similarity

    def value_batch_tensors(self) -> Union[tuple, dict]:
        """Defines a batch of data for the value network"""
        states, returns = super().value_batch_tensors()
        states['action'] = self.memory.actions

        speed = utils.to_tensor(self.env.info_buffer['speed'], expand_axis=-1) / 100.0
        similarity = utils.to_tensor(self.env.info_buffer['similarity'], expand_axis=-1)

        if speed.shape[0] >= returns.shape[0]:
            speed = speed[:returns.shape[0]]
            similarity = similarity[:returns.shape[0]]
        else:
            n = returns.shape[0] - speed.shape[0]
            speed = tf.concat([speed, tf.zeros((n, 1))], axis=0)
            similarity = tf.concat([similarity, tf.zeros((n, 1))], axis=0)

        return states, returns, speed, similarity

    def get_policy_gradients(self, batch):
        states, advantages, log_probabilities, speed, similarity = batch

        if self.should_update_dynamics:
            with tf.GradientTape(persistent=True) as tape:
                # dynamics prediction
                dynamics_out = self.network.dynamics_predict_train(states)
                new_batch = (dynamics_out, advantages, log_probabilities, speed, similarity)

                # policy prediction
                loss = self.policy_objective(batch=new_batch)

            # get gradients with respect to policy, and dynamics
            policy_gradients = tape.gradient(loss, self.network.policy.trainable_variables)
            dynamics_gradients = tape.gradient(loss, self.network.dynamics.trainable_variables)

            del tape
            return loss, dict(policy=policy_gradients, dynamics=dynamics_gradients)
        else:
            dynamics_out = self.network.dynamics_predict_train(states)
            new_batch = (dynamics_out, advantages, log_probabilities, speed, similarity)

            return super().get_policy_gradients(batch=new_batch)

    def apply_policy_gradients(self, gradients):
        if isinstance(gradients, dict):
            assert self.should_update_dynamics

            grads = self.apply_dynamics_gradients(gradients=gradients['dynamics'])
            super().apply_policy_gradients(gradients=gradients['policy'])

            self.log(gradients_norm_dynamics=[tf.norm(g) for g in grads])
        else:
            super().apply_policy_gradients(gradients)

    def apply_dynamics_gradients(self, gradients):
        self.dynamics_optimizer.apply_gradients(zip(gradients, self.network.dynamics.trainable_variables))
        return gradients

    @tf.function
    def policy_predict(self, inputs: dict) -> dict:
        return self.network.policy(inputs, training=True)

    def policy_objective(self, batch):
        states, advantages, old_log_prob, true_speed, true_similarity = batch
        policy = self.policy_predict(states)

        log_prob = policy['log_prob']
        entropy = tf.reduce_mean(policy['entropy'])
        speed = policy['speed']
        similarity = policy['similarity']

        # Entropy
        entropy_penalty = self.entropy_strength() * entropy

        # Compute the probability ratio between the current and old policy
        ratio = tf.math.exp(log_prob - old_log_prob)
        ratio = tf.reduce_mean(ratio, axis=1)  # mean over per-action ratio

        # Compute the clipped ratio times advantage
        clip_value = self.clip_ratio()
        min_adv = tf.where(advantages > 0.0, x=(1.0 + clip_value) * advantages, y=(1.0 - clip_value) * advantages)

        # aux losses
        speed_loss = 0.5 * tf.reduce_mean(losses.MSE(y_true=true_speed, y_pred=speed))
        similarity_loss = 0.5 * tf.reduce_mean(losses.MSE(y_true=true_similarity, y_pred=similarity))

        # total loss
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, min_adv))
        total_loss = policy_loss - entropy_penalty + speed_loss + similarity_loss

        # Log stuff
        self.log(ratio=tf.reduce_mean(ratio), log_prob=tf.reduce_mean(log_prob), entropy=entropy,
                 entropy_coeff=self.entropy_strength.value, ratio_clip=clip_value, loss_speed_policy=speed_loss,
                 loss_policy=policy_loss, loss_entropy=entropy_penalty, speed_pi=tf.reduce_mean(speed),
                 loss_similarity_policy=similarity_loss, similarity_pi=tf.reduce_mean(similarity))

        return total_loss

    def get_value_gradients(self, batch):
        states, returns, speed, similarity = batch

        if self.should_update_dynamics:
            with tf.GradientTape(persistent=True) as tape:
                # dynamics prediction
                dynamics_out = self.network.dynamics_predict_train(states)
                new_batch = (dynamics_out, returns, speed, similarity)

                # value prediction
                loss = self.value_objective(batch=new_batch)

            # get gradients with respect to value, and dynamics
            value_gradients = tape.gradient(loss, self.network.value.trainable_variables)
            dynamics_gradients = tape.gradient(loss, self.network.dynamics.trainable_variables)

            del tape
            return loss, dict(value=value_gradients, dynamics=dynamics_gradients)
        else:
            dynamics_out = self.network.dynamics_predict_train(states)
            new_batch = (dynamics_out, returns, speed, similarity)

            return super().get_value_gradients(batch=new_batch)

    def apply_value_gradients(self, gradients):
        if isinstance(gradients, dict):
            assert self.should_update_dynamics

            grads = self.apply_dynamics_gradients(gradients=gradients['dynamics'])
            super().apply_value_gradients(gradients=gradients['value'])

            self.log(gradients_norm_dynamics_v=[tf.norm(g) for g in grads])
        else:
            super().apply_value_gradients(gradients)

    @tf.function
    def value_predict(self, inputs: dict) -> dict:
        return self.network.value(inputs, training=True)

    def value_objective(self, batch):
        states, returns, true_speed, true_similarity = batch
        prediction = self.value_predict(states)
        values, speed, similarity = prediction['value'], prediction['speed'], prediction['similarity']

        # compute normalized `value loss`:
        base_loss = tf.reduce_mean(losses.MSE(y_true=returns[:, 0], y_pred=values[:, 0]))
        exp_loss = tf.reduce_mean(losses.MSE(y_true=returns[:, 1], y_pred=values[:, 1]))
        value_loss = (0.25 * base_loss) + (exp_loss / (self.network.exp_scale ** 2))

        # auxiliary losses:
        speed_loss = tf.reduce_mean(losses.MSE(y_true=true_speed, y_pred=speed))
        similarity_loss = tf.reduce_mean(losses.MSE(y_true=true_similarity, y_pred=similarity))

        self.log(speed_v=tf.reduce_mean(speed), similarity_v=tf.reduce_mean(similarity),
                 loss_v=value_loss, loss_speed_value=speed_loss, loss_similarity_value=similarity_loss)

        return (value_loss + speed_loss + similarity_loss) * 0.25

    def log_actions(self, **kwargs):
        for tag, actions in kwargs.items():
            for action in actions:
                for i, a in enumerate(action):
                    self.log(**{f'{tag}-{i}': tf.squeeze(a)})

    def convert_actions(self, actions):
        return self.env.to_discrete(actions)

    @tf.function
    def map_states(self, fn: callable, states: dict):
        return tf.map_fn(fn=fn, elems=states)

    @staticmethod
    def convert_command(commands):
        def fn(c):
            i = tf.argmax(c)

            if i == 0:
                # left
                return tf.constant([1.0, 0.0, 0.0], dtype=tf.float32)

            if i == 1:
                # right
                return tf.constant([0.0, 0.0, 1.0], dtype=tf.float32)

            # straight or follow lane
            return tf.constant([0.0, 1.0, 0.0], dtype=tf.float32)

        return tf.map_fn(fn, elems=commands)

    def get_memory(self):
        return CARLAMemory(state_spec=self.state_spec, num_actions=self.num_actions,
                           time_horizon=self.env.time_horizon)

    def preprocess(self):
        """Augmentation function used during the reinforcement learning phase"""
        return self.augment()

    def augment(self):
        """Augmentation function used during the imitation learning phase"""
        alpha = self.aug_intensity
        seed = self.seed

        def prepare(states: list) -> dict:
            tmp = {k: [] for k in states[0].keys()}

            for state in states:
                for k, v in state.items():
                    tmp[k].append(v)

            for k, v in tmp.items():
                tmp[k] = tf.stack(tf.cast(v, dtype=tf.float32), axis=0)

            return {f'state_{k}': v for k, v in tmp.items()}

        # @tf.function
        def augment_fn(states: list):
            state = prepare(states)
            image = utils.to_float(state['state_image'])

            if alpha > 0.0:
                # Color Jitter:
                if aug.tf_chance(seed=seed) < alpha:
                    image = rl.augmentations.simclr.color_jitter(image, strength=alpha, seed=seed)

                # blur
                if aug.tf_chance(seed=seed) < 0.25 * alpha:
                    blur_size = 3 if aug.tf_chance(seed=seed) >= 0.5 else 5
                    image = aug.tf_gaussian_blur(image, size=blur_size, seed=seed)

                # noise
                if aug.tf_chance(seed=seed) < 0.2 * alpha:
                    image = aug.tf_salt_and_pepper_batch(image, amount=0.1)

                if aug.tf_chance(seed=seed) < 0.33 * alpha:
                    image = aug.tf_gaussian_noise_batch(image, amount=0.10, std=0.075, seed=seed)

                image = aug.tf_normalize_batch(image)

                # cutout
                if aug.tf_chance(seed=seed) < 0.15 * alpha:
                    image = aug.tf_cutout_batch(image, size=6, seed=seed)

                # coarse dropout
                if aug.tf_chance(seed=seed) < 0.15 * alpha:
                    image = aug.tf_coarse_dropout_batch(image, size=81, amount=0.04, seed=seed)

            state['state_image'] = image
            return state

        return augment_fn

    def load_weights(self):
        print('loading weights...')
        self.network.load_weights(full=self.load_full)


class CARLAMemory(PPOMemory):
    def __init__(self, state_spec: dict, num_actions: int, time_horizon: int):
        super().__init__(state_spec, num_actions)
        self.time_horizon = time_horizon

        # consider `time_horizon` for states:
        if self.simple_state:
            self.states = tf.zeros(shape=(0, self.time_horizon) + state_spec.get('state'), dtype=tf.float32)
        else:
            for name, shape in state_spec.items():
                self.states[name] = tf.zeros(shape=(0, self.time_horizon) + shape, dtype=tf.float32)
