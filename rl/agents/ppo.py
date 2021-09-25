"""Proximal Policy Optimization Agent"""

import os
import gym
import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import random

from typing import Union

from rl import utils
from rl.agents.agents import Agent
from rl.parameters import DynamicParameter
from rl.networks.networks import PPONetwork

from tensorflow.keras import losses
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class PPOAgent(Agent):
    # TODO: dynamic-parameters: gamma, lambda, opt_steps, update_freq?, polyak?, clip_norm
    # TODO: debug each action separately
    # TODO: RNN support
    def __init__(self, *args, policy_lr: Union[float, LearningRateSchedule, DynamicParameter] = 1e-3, gamma=0.99,
                 lambda_=0.95, value_lr: Union[float, LearningRateSchedule, DynamicParameter] = 3e-4, load=False,
                 optimization_steps=(1, 1), name='ppo-agent', optimizer='adam', clip_norm=(1.0, 1.0),
                 clip_ratio: Union[float, LearningRateSchedule, DynamicParameter] = 0.2, seed_regularization=False,
                 entropy_regularization: Union[float, LearningRateSchedule, DynamicParameter] = 0.0,
                 network: Union[dict, PPONetwork] = None, update_frequency=1, polyak=1.0, repeat_action=1,
                 advantage_scale: Union[float, LearningRateSchedule, DynamicParameter] = 2.0, **kwargs):
        assert 0.0 < polyak <= 1.0
        assert repeat_action >= 1

        super().__init__(*args, name=name, **kwargs)

        self.memory: PPOMemory = None
        self.gamma = gamma
        self.lambda_ = lambda_
        self.repeat_action = repeat_action
        self.adv_scale = DynamicParameter.create(value=advantage_scale)

        if seed_regularization:
            def _seed_regularization():
                seed = random.randint(a=0, b=2**32 - 1)
                self.set_random_seed(seed)

            self.seed_regularization = _seed_regularization
            self.seed_regularization()
        else:
            self.seed_regularization = lambda: None

        # Entropy regularization
        self.entropy_strength = DynamicParameter.create(value=entropy_regularization)

        # Ratio clipping
        if isinstance(clip_ratio, float):
            assert clip_ratio >= 0.0

        self.clip_ratio = DynamicParameter.create(value=clip_ratio)

        # Action space
        self._init_action_space()

        print('state_spec:', self.state_spec)
        print('action_shape:', self.num_actions)
        print('distribution:', self.distribution_type)

        # Gradient clipping:
        self._init_gradient_clipping(clip_norm)

        # Networks & Loading
        self.weights_path = dict(policy=os.path.join(self.base_path, 'policy_net'),
                                 value=os.path.join(self.base_path, 'value_net'))

        if isinstance(network, dict):
            network_class = network.pop('network', PPONetwork)

            if network_class is PPONetwork:
                # policy/value-specific arguments
                policy_args = network.pop('policy', {})
                value_args = network.pop('value', policy_args)

                # common arguments
                for k, v in network.items():
                    if k not in policy_args:
                        policy_args[k] = v

                    if k not in value_args:
                        value_args[k] = v

                self.network = network_class(agent=self, policy=policy_args, value=value_args, **network)
            else:
                self.network = network_class(agent=self, **network)
        else:
            self.network = PPONetwork(agent=self, policy={}, value={})

        # Optimization
        self.update_frequency = update_frequency
        self.policy_lr = DynamicParameter.create(value=policy_lr)
        self.value_lr = DynamicParameter.create(value=value_lr)
        self.optimization_steps = dict(policy=optimization_steps[0], value=optimization_steps[1])

        self.policy_optimizer = utils.get_optimizer_by_name(optimizer, learning_rate=self.policy_lr)
        self.value_optimizer = utils.get_optimizer_by_name(optimizer, learning_rate=self.value_lr)

        self.should_polyak_average = polyak < 1.0
        self.polyak_coeff = polyak

        if load:
            self.load()

    def _init_gradient_clipping(self, clip_norm: Union[tuple, float, None]):
        if clip_norm is None:
            self.should_clip_policy_grads = False
            self.should_clip_value_grads = False

        elif isinstance(clip_norm, float):
            assert clip_norm > 0.0
            self.should_clip_policy_grads = True
            self.should_clip_value_grads = True

            self.grad_norm_policy = clip_norm
            self.grad_norm_value = clip_norm
        else:
            assert isinstance(clip_norm, tuple)

            if clip_norm[0] is None:
                self.should_clip_policy_grads = False
            else:
                assert isinstance(clip_norm[0], float)
                assert clip_norm[0] > 0.0

                self.should_clip_policy_grads = True
                self.grad_norm_policy = tf.constant(clip_norm[0], dtype=tf.float32)

            if clip_norm[1] is None:
                self.should_clip_value_grads = False
            else:
                assert isinstance(clip_norm[1], float)
                assert clip_norm[1] > 0.0

                self.should_clip_value_grads = True
                self.grad_norm_value = tf.constant(clip_norm[1], dtype=tf.float32)

    # TODO: handle complex action spaces (make use of Agent.action_spec)
    def _init_action_space(self):
        action_space = self.env.action_space

        if isinstance(action_space, gym.spaces.Box):
            self.num_actions = action_space.shape[0]

            # continuous:
            if action_space.is_bounded():
                self.distribution_type = 'beta'

                self.action_low = tf.constant(action_space.low, dtype=tf.float32)
                self.action_high = tf.constant(action_space.high, dtype=tf.float32)
                self.action_range = tf.constant(action_space.high - action_space.low,
                                                dtype=tf.float32)

                self.convert_action = lambda a: (a * self.action_range + self.action_low)[0].numpy()
            else:
                self.distribution_type = 'gaussian'
                self.convert_action = lambda a: a[0].numpy()
        else:
            # discrete:
            self.distribution_type = 'categorical'

            if isinstance(action_space, gym.spaces.MultiDiscrete):
                # make sure all discrete components of the space have the same number of classes
                assert np.all(action_space.nvec == action_space.nvec[0])

                self.num_actions = action_space.nvec.shape[0]
                self.num_classes = action_space.nvec[0] + 1  # to include the last class, i.e. 0 to K (not 0 to k-1)
                self.convert_action = lambda a: tf.cast(a[0], dtype=tf.int32).numpy()
            else:
                self.num_actions = 1
                self.num_classes = action_space.n
                self.convert_action = lambda a: tf.cast(tf.squeeze(a), dtype=tf.int32).numpy()

    def act(self, state, *args, **kwargs):
        action = self.network.act(inputs=state)
        return self.convert_action(action)

    def predict(self, state, *args, **kwargs):
        return self.network.predict(inputs=state)

    def update(self):
        t0 = time.time()
        self.seed_regularization()

        # Prepare data:
        value_batches = self.get_value_batches()
        policy_batches = self.get_policy_batches()

        # Policy network optimization:
        for opt_step in range(self.optimization_steps['policy']):
            for data_batch in policy_batches:
                self.seed_regularization()
                total_loss, policy_grads = self.get_policy_gradients(data_batch)

                self.update_policy(policy_grads)

                if isinstance(policy_grads, dict):
                    policy_grads = policy_grads['policy']

                self.log(loss_total=total_loss, lr_policy=self.policy_lr.value,
                         gradients_norm_policy=[tf.norm(gradient) for gradient in policy_grads])

        # Value network optimization:
        for _ in range(self.optimization_steps['value']):
            for data_batch in value_batches:
                self.seed_regularization()
                value_loss, value_grads = self.get_value_gradients(data_batch)

                self.update_value(value_grads)

                if isinstance(value_grads, dict):
                    value_grads = value_grads['value']

                self.log(loss_value=value_loss, lr_value=self.value_lr.value,
                         gradients_norm_value=[tf.norm(gradient) for gradient in value_grads])

        print(f'Update took {round(time.time() - t0, 3)}s')

    def get_policy_gradients(self, batch):
        with tf.GradientTape() as tape:
            loss = self.policy_objective(batch)

        gradients = tape.gradient(loss, self.network.policy.trainable_variables)
        return loss, gradients

    def update_policy(self, gradients) -> (list, bool):
        return self.apply_policy_gradients(gradients), True

    def apply_policy_gradients(self, gradients):
        if self.should_clip_policy_grads:
            gradients = utils.clip_gradients(gradients, norm=self.grad_norm_policy)

        if self.should_polyak_average:
            old_weights = self.network.policy.get_weights()
            self.network.update_old_policy(old_weights)

            self.policy_optimizer.apply_gradients(zip(gradients, self.network.policy.trainable_variables))
            utils.polyak_averaging(self.network.policy, old_weights, alpha=self.polyak_coeff)
        else:
            self.network.update_old_policy()
            self.policy_optimizer.apply_gradients(zip(gradients, self.network.policy.trainable_variables))

        return gradients

    def get_value_gradients(self, batch):
        with tf.GradientTape() as tape:
            loss = self.value_objective(batch)

        gradients = tape.gradient(loss, self.network.value.trainable_variables)
        return loss, gradients

    def update_value(self, gradients) -> (list, bool):
        return self.apply_value_gradients(gradients), True

    def apply_value_gradients(self, gradients):
        if self.should_clip_value_grads:
            gradients = utils.clip_gradients(gradients, norm=self.grad_norm_value)

        if self.should_polyak_average:
            old_weights = self.network.value.get_weights()
            self.value_optimizer.apply_gradients(zip(gradients, self.network.value.trainable_variables))
            utils.polyak_averaging(self.network.value, old_weights, alpha=self.polyak_coeff)
        else:
            self.value_optimizer.apply_gradients(zip(gradients, self.network.value.trainable_variables))

        return gradients

    def value_batch_tensors(self) -> Union[tuple, dict]:
        """Defines which data to use in `get_value_batches()`"""
        return self.memory.states, self.memory.returns

    def policy_batch_tensors(self) -> Union[tuple, dict]:
        """Defines which data to use in `get_policy_batches()`"""
        return self.memory.states, self.memory.advantages, self.memory.actions, self.memory.log_probabilities

    def get_value_batches(self):
        """Computes batches of data for updating the value network"""
        return utils.data_to_batches(tensors=self.value_batch_tensors(), batch_size=self.batch_size,
                                     drop_remainder=self.drop_batch_remainder, skip=self.skip_count,
                                     shuffle=True, shuffle_batches=False, num_shards=self.obs_skipping)

    def get_policy_batches(self):
        """Computes batches of data for updating the policy network"""
        return utils.data_to_batches(tensors=self.policy_batch_tensors(), batch_size=self.batch_size,
                                     drop_remainder=self.drop_batch_remainder, skip=self.skip_count,
                                     num_shards=self.obs_skipping, shuffle=self.shuffle,
                                     shuffle_batches=self.shuffle_batches)

    @tf.function
    def value_objective(self, batch):
        states, returns = batch[:2]
        values = self.network.value(states, training=True)

        base_loss = tf.reduce_mean(losses.MSE(y_true=returns[:, 0], y_pred=values[:, 0]))
        exp_loss = tf.reduce_mean(losses.MSE(y_true=returns[:, 1], y_pred=values[:, 1]))

        # normalized loss by (0.25 = 1/2^2) and (1/k^2)
        return 0.5 * (0.25 * base_loss + exp_loss / (self.network.exp_scale ** 2))

    def policy_objective(self, batch):
        """PPO-Clip Objective"""
        states, advantages, actions, old_log_probabilities = batch[:4]
        new_policy: tfp.distributions.Distribution = self.network.policy(states, training=True)

        # TODO: probable bug -> "self.num_actions == 1"??
        if self.distribution_type == 'categorical' and self.num_actions == 1:
            batch_size = tf.shape(actions)[0]
            actions = tf.reshape(actions, shape=batch_size)

            new_log_prob = new_policy.log_prob(actions)
            new_log_prob = tf.reshape(new_log_prob, shape=(batch_size, self.num_actions))
        else:
            # round samples (actions) before computing density:
            # motivation: https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Beta
            actions = tf.clip_by_value(actions, utils.EPSILON, 1.0 - utils.EPSILON)
            new_log_prob = new_policy.log_prob(actions)

        kl_divergence = utils.kl_divergence(old_log_probabilities, new_log_prob)
        kl_divergence = tf.reduce_mean(kl_divergence)

        # Entropy
        entropy = tf.reduce_mean(new_policy.entropy())
        entropy_penalty = self.entropy_strength() * entropy

        # Compute the probability ratio between the current and old policy
        ratio = tf.math.exp(new_log_prob - old_log_probabilities)
        ratio = tf.reduce_mean(ratio, axis=1)  # mean over per-action ratio

        # Compute the clipped ratio times advantage
        clip_value = self.clip_ratio()
        clipped_ratio = tf.clip_by_value(ratio, clip_value_min=1.0 - clip_value, clip_value_max=1.0 + clip_value)

        # Source: https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/ppo/ppo.py#L201
        min_adv = tf.where(advantages > 0.0, x=(1.0 + clip_value) * advantages, y=(1.0 - clip_value) * advantages)

        # Loss = min { ratio * A, clipped_ratio * A } + entropy_term
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, min_adv))
        # policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
        total_loss = policy_loss - entropy_penalty

        # Log stuff
        self.log(ratio=tf.reduce_mean(ratio), log_prob=tf.reduce_mean(new_log_prob), entropy=entropy,
                 entropy_coeff=self.entropy_strength.value, ratio_clip=clip_value, kl_divergence=kl_divergence,
                 loss_policy=policy_loss.numpy(), loss_entropy=entropy_penalty.numpy(),
                 # adv_diff=clipped_ratio - min_adv, adv_ratio=ratio * advantages, adv_min=min_adv,
                 # adv_minimum=tf.minimum(ratio * advantages, min_adv), adv_min_diff=ratio * advantages - min_adv
                 )

        return total_loss, kl_divergence

    def collect(self, episodes: int, timesteps: int, render=True, record_threshold=0.0, seeds=None, close=True):
        import random
        sample_seed = False

        if isinstance(seeds, int):
            self.set_random_seed(seed=seeds)
        elif isinstance(seeds, list):
            sample_seed = True

        for episode in range(1, episodes + 1):
            if sample_seed:
                self.set_random_seed(seed=random.choice(seeds))

            self.reset()
            episode_reward = 0.0
            memory = PPOMemory(state_spec=self.state_spec, num_actions=self.num_actions)

            state = self.env.reset()
            state = utils.to_tensor(state)

            if isinstance(state, dict):
                state = {f'state_{k}': v for k, v in state.items()}

            for t in range(1, timesteps + 1):
                if render:
                    self.env.render()

                action, log_prob, value = self.network.act2(state)
                next_state, reward, done, _ = self.env.step(self.convert_action(action))
                episode_reward += reward

                self.log(actions=action, rewards=reward, values=value, log_probs=log_prob)

                memory.append(state, action, reward, value, log_prob)
                state = utils.to_tensor(next_state)

                if isinstance(state, dict):
                    state = {f'state_{k}': v for k, v in state.items()}

                if done or (t == timesteps):
                    print(f'Episode {episode} terminated after {t} timesteps with reward {episode_reward}.')

                    last_value = self.network.predict_last_value(state, is_terminal=done)
                    memory.end_trajectory(last_value)
                    break

            self.log(evaluation_reward=episode_reward)
            self.write_summaries()

            if episode_reward >= record_threshold:
                memory.serialize(episode, save_path=self.traces_dir)

        if close:
            self.env.close()

    def imitate(self, epochs=1, batch_size: Union[None, int] = None, shuffle_batches=False, shuffle_data=False,
                close=True, seed=None):
        """Learn from experience traces collected by 'collect'"""
        batch_size = self.batch_size if batch_size is None else batch_size
        self.set_random_seed(seed)

        for epoch in range(1, epochs + 1):
            for i, trace in enumerate(utils.load_traces(self.traces_dir, shuffle=True)):
                t0 = time.time()
                self.reset()
                trace = utils.unpack_trace(trace, unpack=False)

                states = utils.to_tensor(trace['state'])[0]
                actions = utils.to_float(trace['action'])
                log_probs = utils.to_float(trace['log_prob'])
                rewards = utils.to_float(trace['reward'])
                values = utils.to_float(trace['value'])

                data = utils.data_to_batches((states, actions, log_probs, rewards, values), batch_size=batch_size,
                                             shuffle_batches=shuffle_batches, shuffle=shuffle_data, seed=seed,
                                             skip=self.skip_count, num_shards=self.obs_skipping,
                                             drop_remainder=self.drop_batch_remainder)
                for batch in data:
                    states, actions, log_probs, rewards, values = batch

                    # TODO: refactor
                    memory = PPOMemory(state_spec=self.state_spec, num_actions=self.num_actions)
                    memory.rewards = rewards
                    memory.values = values
                    memory.compute_returns(discount=self.gamma)
                    memory.compute_advantages(self.gamma, self.lambda_)

                    # update policy:
                    loss_policy, policy_grads = self.get_policy_gradients(batch=(states, memory.advantages,
                                                                                    actions, log_probs))
                    self.apply_policy_gradients(policy_grads)

                    # update value:
                    loss_value, value_grads = self.get_value_gradients(batch=(states, memory.returns))
                    self.apply_value_gradients(value_grads)

                    self.log(loss_policy=loss_policy, loss_value=loss_value, advantages=memory.advantages)

                self.write_summaries()
                print(f'[{epoch}] Trace-{i} took {round(time.time() - t0, 3)}s.')

        if close:
            self.env.close()

    def learn(self, episodes: int, timesteps: int, save_every: Union[bool, str, int] = False,
              render_every: Union[bool, str, int] = False, close=True):
        assert episodes % self.update_frequency == 0

        if (save_every is False) or (save_every is None):
            save_every = episodes + 1
        elif save_every is True:
            save_every = 1
        elif save_every == 'end':
            save_every = episodes
        else:
            assert episodes % save_every == 0

        if render_every is False:
            render_every = episodes + 1
        elif render_every is True:
            render_every = 1

        try:
            self.memory = self.get_memory()

            for episode in range(1, episodes + 1):
                self.seed_regularization()
                self.on_episode_start()

                preprocess_fn = self.preprocess()
                self.reset()

                state = self.env.reset()
                episode_reward = 0.0
                t0 = time.time()
                render = episode % render_every == 0

                for t in range(1, timesteps + 1):
                    if render:
                        self.env.render()

                    if isinstance(state, dict):
                        state = {f'state_{k}': v for k, v in state.items()}

                    state = preprocess_fn(state)
                    state = utils.to_tensor(state)

                    if (t + 1) % 10 == 0:
                        self.log(image_state=state['state_image'])

                    # Agent prediction
                    action, mean, std, log_prob, value = self.predict(state)
                    action_env = self.convert_action(action)

                    # Environment step
                    for _ in range(self.repeat_action):
                        next_state, reward, done, _ = self.env.step(action_env)
                        episode_reward += reward

                        if done:
                            break

                    self.log(actions=action, action_env=action_env, rewards=reward,
                             distribution_mean=mean, distribution_std=std)

                    self.memory.append(state, action, reward, value, log_prob)
                    state = next_state

                    # check whether a termination (terminal state or end of a transition) is reached:
                    if done or (t == timesteps):
                        print(f'Episode {episode} terminated after {t} timesteps in {round((time.time() - t0), 3)}s ' +
                              f'with reward {round(episode_reward, 3)}.')
                        self.log(timestep=t)

                        if isinstance(state, dict):
                            state = {f'state_{k}': v for k, v in state.items()}

                        state = preprocess_fn(state)
                        state = utils.to_tensor(state)

                        last_value = self.network.predict_last_value(state, timestep=(t + 1) / timesteps,
                                                                     is_terminal=done)
                        self.end_episode(last_value, append=self.update_frequency > 1)
                        break

                if episode % self.update_frequency == 0:
                    self.update()
                    self.memory.delete()
                    self.memory = self.get_memory()

                elif self.update_frequency > 1:
                    # remove last `reward` and `value` to avoid shape issue when building the batch for update()
                    self.memory.rewards = self.memory.rewards[:-1]
                    self.memory.values = self.memory.values[:-1]

                self.log(episode_rewards=episode_reward)
                self.write_summaries()

                if self.should_record:
                    self.record(episode)

                self.on_episode_end()

                if episode % save_every == 0:
                    self.save()
        finally:
            if close:
                print('closing...')
                self.env.close()

    def get_memory(self):
        """Instantiate the agent's memory; easy to subclass"""
        return PPOMemory(state_spec=self.state_spec, num_actions=self.num_actions)

    def end_episode(self, last_value, append=False):
        """Used during learning `learn(...)` to terminate an episode"""
        self.memory.end_trajectory(last_value)
        returns = self.memory.compute_returns(discount=self.gamma, append=append)
        values, advantages = self.memory.compute_advantages(self.gamma, self.lambda_,
                                                            scale=self.adv_scale(), append=append)
        self.memory.update_index(append=append)

        self.log(returns=returns, advantages=advantages, values=values, advantage_scale=self.adv_scale.value,
                 returns_base=self.memory.returns[:, 0], returns_exp=self.memory.returns[:, 1],
                 values_base=self.memory.values[:, 0], values_exp=self.memory.values[:, 1],
                 returns_minus_values=returns - values[:-1], advantages_normalized=self.memory.advantages)

    def record(self, episode: int):
        self.memory.serialize(episode, save_path=self.traces_dir)

    def summary(self):
        self.network.summary()

    def save_weights(self):
        print('saving weights...')
        self.network.save_weights()

    def load_weights(self):
        print('loading weights...')
        self.network.load_weights()

    def save_config(self):
        print('save config')
        self.update_config(policy_lr=self.policy_lr.serialize(), value_lr=self.value_lr.serialize(),
                           adv_scale=self.adv_scale.serialize(),
                           entropy_strength=self.entropy_strength.serialize(), clip_ratio=self.clip_ratio.serialize())
        super().save_config()

    def load_config(self):
        print('load config')
        super().load_config()

        self.policy_lr.load(config=self.config.get('policy_lr', {}))
        self.value_lr.load(config=self.config.get('value_lr', {}))
        self.adv_scale.load(config=self.config.get('adv_scale', {}))
        self.entropy_strength.load(config=self.config.get('entropy_strength', {}))
        self.clip_ratio.load(config=self.config.get('clip_ratio', {}))

    def reset(self):
        super().reset()
        self.network.reset()

    def on_episode_end(self):
        super().on_episode_end()
        self.policy_lr.on_episode()
        self.value_lr.on_episode()
        self.adv_scale.on_episode()


class PPOMemory:
    """Recent memory used in PPOAgent"""

    # TODO: define what to store from a specification (dict: str -> (shape, dtype))
    def __init__(self, state_spec: dict, num_actions: int):
        self.index = 0

        if list(state_spec.keys()) == ['state']:
            # Simple state-space
            self.states = tf.zeros(shape=(0,) + state_spec.get('state'), dtype=tf.float32)
            self.simple_state = True
        else:
            # Complex state-space
            self.states = dict()
            self.simple_state = False

            for name, shape in state_spec.items():
                self.states[name] = tf.zeros(shape=(0,) + shape, dtype=tf.float32)

        self.rewards = tf.zeros(shape=(0,), dtype=tf.float32)
        self.values = tf.zeros(shape=(0, 2), dtype=tf.float32)
        self.actions = tf.zeros(shape=(0, num_actions), dtype=tf.float32)
        self.log_probabilities = tf.zeros(shape=(0, num_actions), dtype=tf.float32)
        self.timesteps = tf.zeros(shape=(0,), dtype=tf.float32)

        self.returns = None
        self.advantages = None

    def __len__(self):
        return self.actions.shape[0]

    def delete(self):
        if self.simple_state:
            del self.states
        else:
            for k in self.states.keys():
                self.states[k] = None

            del self.states

        del self.rewards
        del self.values
        del self.actions
        del self.log_probabilities
        del self.timesteps
        del self.returns
        del self.advantages

    # TODO: use kwargs to define what to append and where to store?
    def append(self, state, action, reward, value, log_prob):
        if self.simple_state:
            self.states = tf.concat([self.states, state], axis=0)
        else:
            assert isinstance(state, dict)

            for k, v in state.items():
                self.states[k] = tf.concat([self.states[k], v], axis=0)

        self.actions = tf.concat([self.actions, tf.cast(action, dtype=tf.float32)], axis=0)
        self.rewards = tf.concat([self.rewards, [reward]], axis=0)
        self.values = tf.concat([self.values, value], axis=0)
        self.log_probabilities = tf.concat([self.log_probabilities, log_prob], axis=0)

    def end_trajectory(self, last_value: tf.Tensor):
        """Terminates the current trajectory by adding the value of the terminal state"""
        value = last_value[:, 0] * tf.pow(10.0, last_value[:, 1])

        self.rewards = tf.concat([self.rewards, value], axis=0)
        self.values = tf.concat([self.values, last_value], axis=0)

    def compute_returns(self, discount: float, append=False):
        """Computes the returns, also called rewards-to-go"""
        returns = utils.rewards_to_go(self.rewards[self.index:], discount=discount)
        returns = utils.to_float(returns)

        new_returns = tf.map_fn(fn=utils.decompose_number, elems=returns, dtype=(tf.float32, tf.float32))
        new_returns = tf.stack(new_returns, axis=1)

        if (self.returns is None) or (not append):
            self.returns = new_returns
        elif append is True:
            self.returns = tf.concat([self.returns, new_returns], axis=0)

        return returns

    def compute_advantages(self, gamma: float, lambda_: float, scale=2.0, append=False):
        """Computes the advantages using generalized-advantage estimation"""
        # value = base * 10^exponent
        values = self.values[self.index:, 0] * tf.pow(10.0, self.values[self.index:, 1])

        advantages = utils.gae(self.rewards[self.index:], values=values, gamma=gamma, lambda_=lambda_, normalize=False)
        new_advantages = utils.tf_sp_norm(advantages) * scale

        if (self.advantages is None) or (not append):
            self.advantages = new_advantages
        elif append is True:
            self.advantages = tf.concat([self.advantages, new_advantages], axis=0)

        return values, advantages

    def update_index(self, append=False):
        if append:
            self.index = self.rewards.shape[0] - 1
        else:
            self.index = self.rewards.shape[0]

    def serialize(self, episode: int, save_path: str):
        """Writes to file (npz - numpy compressed format) all the transitions (state, reward, action) collected so
           far.
        """
        # Trace's file path:
        filename = f'trace-{episode}-{time.strftime("%Y%m%d-%H%M%S")}.npz'
        trace_path = os.path.join(save_path, filename)

        # Select data to save
        buffer = dict(reward=self.rewards, action=self.actions, value=self.values, log_prob=self.log_probabilities)

        if self.simple_state:
            buffer['state'] = self.states
        else:
            for key, value in self.states.items():
                buffer[key] = value

        # Save buffer
        np.savez_compressed(file=trace_path, **buffer)
        print(f'Traces "{filename}" saved.')
