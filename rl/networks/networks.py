
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from typing import List, Union, Dict

from rl import utils


class Network:
    def __init__(self, agent):
        self.agent = agent

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def act(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        pass

    def trainable_variables(self):
        raise NotImplementedError

    def set_weights(self, weights):
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError

    def load_weights(self):
        raise NotImplementedError

    def save_weights(self):
        raise NotImplementedError

    def summary(self):
        pass

    def _get_input_layers(self, include_actions=False) -> Dict[str, Input]:
        """Transforms arbitrary complex state-spaces (and, optionally, action-spaces) as input layers"""
        input_layers = dict()

        for name, shape in self.agent.state_spec.items():
            # if self.agent.drop_batch_remainder:
            #     layer = Input(shape=shape, batch_size=self.agent.batch_size, dtype=tf.float32, name=name)
            # else:
            #     layer = Input(shape=shape, dtype=tf.float32, name=name)

            layer = Input(shape=shape, dtype=tf.float32, name=name)
            input_layers[name] = layer

        # TODO: bugged for discrete actions (requires '-1' in shape)
        if include_actions:
            for name, shape in self.agent.action_spec.items():
                layer = Input(shape=shape, dtype=tf.float32, name=name)
                input_layers[name] = layer

        return input_layers

    @staticmethod
    def _clip_actions(actions):
        """Clips actions to prevent numerical instability when computing (log-)probabilities.
           - Use for Beta distribution only.
        """
        return tf.clip_by_value(actions, utils.EPSILON, 1.0 - utils.EPSILON)

    def get_distribution_layer(self, distribution: str, layer: Layer) -> tfp.layers.DistributionLambda:
        # Discrete actions:
        if distribution == 'categorical':
            num_actions = self.agent.num_actions
            num_classes = self.agent.num_classes

            logits = Dense(units=num_actions * num_classes, activation='linear', name='logits')(layer)

            if num_actions > 1:
                logits = Reshape((num_actions, num_classes))(logits)
            else:
                logits = tf.expand_dims(logits, axis=0)

            return tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Categorical(logits=t))(logits)

        # Bounded continuous 1-dimensional actions:
        # for activations choice refer to chapter 4 of http://proceedings.mlr.press/v70/chou17a/chou17a.pdf
        if distribution == 'beta':
            num_actions = self.agent.num_actions

            # make a, b > 1, so that the Beta distribution is concave and unimodal (see paper above)
            alpha = Dense(units=num_actions, activation=utils.softplus(1.0 + 1e-2), name='alpha')(layer)
            beta = Dense(units=num_actions, activation=utils.softplus(1.0 + 1e-2), name='beta')(layer)

            return tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Beta(t[0], t[1]))([alpha, beta])

        # Unbounded continuous actions)
        # for activations choice see chapter 4 of http://proceedings.mlr.press/v70/chou17a/chou17a.pdf
        if distribution == 'gaussian':
            num_actions = self.agent.num_actions

            mu = Dense(units=num_actions, activation='linear', name='mu')(layer)
            sigma = Dense(units=num_actions, activation=utils.softplus(1.0 + 1e-2), name='sigma')(layer)

            return tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Normal(loc=t[0], scale=t[1]))([mu, sigma])


# TODO: disentangle policy-net from value-net, so that each of them can be arbitrary subclassed, moreover a
#  Network class can be composed by these policy/value/Q-network classes...
class PPONetwork(Network):
    def __init__(self, agent, policy: dict, value: dict):
        from rl.agents.ppo import PPOAgent
        super().__init__(agent)
        self.agent: PPOAgent

        self.distribution = self.agent.distribution_type

        # policy network
        self.policy = self.policy_network(**policy)
        self.old_policy = self.policy_network(**policy)
        self.update_old_policy()

        # value network
        self.exp_scale = 6.0
        self.value = self.value_network(**value)
        self.last_value = tf.zeros((1, 2), dtype=tf.float32)  # (base, exp)

    @tf.function
    def predict(self, inputs: Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]]):
        policy = self.old_policy(inputs, training=False)
        value = self.value_predict(inputs)

        if self.distribution != 'categorical':
            # round samples (actions) before computing density:
            # motivation: https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Beta
            log_prob = policy.log_prob(tf.clip_by_value(policy, utils.EPSILON, 1.0 - utils.EPSILON))
            mean = policy.mean()
            std = policy.stddev()
        else:
            mean = 0.0
            std = 0.0
            log_prob = policy.log_prob(policy)

        return policy, mean, std, log_prob, value

    # @tf.function
    def policy_predict(self, inputs):
        return self.policy(inputs, training=False)

    @tf.function
    def value_predict(self, inputs):
        return self.value(inputs, training=False)

    def predict_last_value(self, state, timestep: float, is_terminal: bool):
        if is_terminal:
            return self.last_value

        return self.value_predict(state)

    @tf.function
    def act(self, inputs: Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]]):
        action = self.policy(inputs, training=False)
        return action

    @tf.function
    def act2(self, inputs: Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]]):
        policy = self.policy(inputs, training=False)
        value = self.value_predict(inputs)

        if self.distribution != 'categorical':
            log_prob = policy.log_prob(tf.clip_by_value(policy, utils.EPSILON, 1.0 - utils.EPSILON))
        else:
            log_prob = policy.log_prob(policy)

        return policy, log_prob, value

    def policy_layers(self, inputs: Dict[str, Input], **kwargs):
        """Defines the architecture of the policy-network"""
        units = kwargs.get('units', 32)
        num_layers = kwargs.get('num_layers', kwargs.get('layers', 2))  # 'num_layers' or 'layers'
        activation = kwargs.get('activation', tf.nn.swish)
        dropout_rate = kwargs.get('dropout', 0.0)
        linear_units = kwargs.get('linear_units', 0)

        x = Dense(units, activation=activation)(inputs['state'])
        x = LayerNormalization()(x)

        for _ in range(0, num_layers, 2):
            if dropout_rate > 0.0:
                x = Dense(units, activation=activation)(x)
                x = Dropout(rate=dropout_rate)(x)

                x = Dense(units, activation=activation)(x)
                x = Dropout(rate=dropout_rate)(x)
            else:
                x = Dense(units, activation=activation)(x)
                x = Dense(units, activation=activation)(x)

            x = LayerNormalization()(x)

        if linear_units > 0:
            x = Dense(units=linear_units, activation='linear')(x)

        return x

    def value_layers(self, inputs: Dict[str, Input], **kwargs):
        """Defines the architecture of the value-network"""
        return self.policy_layers(inputs, **kwargs)

    def policy_network(self, **kwargs):
        inputs = self._get_input_layers()
        last_layer = self.policy_layers(inputs, **kwargs)
        action = self.get_distribution_layer(distribution=self.distribution, layer=last_layer)

        return Model(list(inputs.values()), outputs=action, name='Policy-Network')

    def value_network(self, **kwargs):
        inputs = self._get_input_layers()
        last_layer = self.value_layers(inputs, **kwargs)

        value = self.value_head(last_layer, **kwargs)
        self.exp_scale = kwargs.get('exponent_scale', self.exp_scale)

        return Model(list(inputs.values()), outputs=value, name='Value-Network')

    def value_head(self, layer: Layer, exponent_scale=6.0, components=1, **kwargs):
        assert components >= 1
        assert exponent_scale > 0.0

        if components == 1:
            base = Dense(units=1, activation=tf.nn.tanh, name='v-base')(layer)
            exp = Dense(units=1, activation=lambda x: exponent_scale * tf.nn.sigmoid(x), name='v-exp')(layer)
        else:
            weights_base = Dense(units=components, activation='softmax', name='w-base')(layer)
            weights_exp = Dense(units=components, activation='softmax', name='w-exp')(layer)

            base = Dense(units=components, activation=tf.nn.tanh, name='v-base')(layer)
            base = utils.tf_dot_product(base, weights_base, axis=1, keepdims=True)

            exp = Dense(units=components, activation=lambda x: exponent_scale * tf.nn.sigmoid(x), name='v-exp')(layer)
            exp = utils.tf_dot_product(exp, weights_exp, axis=1, keepdims=True)

        return concatenate([base, exp], axis=1)

    def gaussian_value_head(self, last_layer: Layer, mixture_components=3, activation=tf.nn.swish, **kwargs):
        num_params = tfp.layers.MixtureNormal.params_size(mixture_components, event_shape=(1,))
        params = Dense(units=num_params, activation=activation, name='value-parameters')(last_layer)

        return tfp.layers.MixtureNormal(mixture_components, event_shape=(1,))(params)

    def load_weights(self):
        self.policy.load_weights(filepath=self.agent.weights_path['policy'], by_name=False)
        self.old_policy.set_weights(self.policy.get_weights())
        self.value.load_weights(filepath=self.agent.weights_path['value'], by_name=False)

    def save_weights(self):
        self.policy.save_weights(filepath=self.agent.weights_path['policy'])
        self.value.save_weights(filepath=self.agent.weights_path['value'])

    def update_old_policy(self, weights=None):
        if weights:
            self.old_policy.set_weights(weights)
        else:
            self.old_policy.set_weights(self.policy.get_weights())

    def summary(self):
        print('==== Policy Network ====')
        self.policy.summary()

        print('\n==== Value Network ====')
        self.value.summary()
