"""Networks architectures for CARLA Agent"""

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from typing import List, Dict, Union

from rl import networks, utils
from rl.networks import Network

from core import architectures as nn


TensorType = Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]]


# -------------------------------------------------------------------------------------------------
# -- SHARED NETWORK
# -------------------------------------------------------------------------------------------------

def linear_combination(inputs, units=32, normalization='batch', name=None):
    if normalization == 'batch':
        x = BatchNormalization()(inputs)
    else:
        x = inputs

    return Dense(units=units, activation='linear', bias_initializer='glorot_uniform', name=name)(x)


def stack(layers: List[Layer], axis=1):
    return tf.stack(layers, axis=axis)


def dynamics_layers(inputs: dict, time_horizon: int, **kwargs):
    """Defined the shared-network architecture, returns its last layer"""
    # process observations (over time):
    image_out = nn.shufflenet_v2(inputs['state_image'], time_horizon, **kwargs.get('shufflenet', {}))
    road_out = nn.feature_net(inputs['state_road'], time_horizon, **kwargs.get('road', dict(normalization=None)))
    vehicle_out = nn.feature_net(inputs['state_vehicle'], time_horizon, **kwargs.get('vehicle', {}), )
    navigation_out = nn.feature_net(inputs['state_navigation'], time_horizon, **kwargs.get('navigation', {}), )

    # use RNNs to aggregate predictions over time:
    args = kwargs.get('rnn')
    image_out = GRU(units=args.get('image'), unroll=True, bias_initializer='glorot_uniform')(stack(image_out))
    road_out = GRU(units=args.get('road'), unroll=True, bias_initializer='glorot_uniform')(stack(road_out))
    vehicle_out = GRU(units=args.get('vehicle'), unroll=True, bias_initializer='glorot_uniform')(stack(vehicle_out))
    navigation_out = GRU(units=args.get('navigation'), unroll=True, bias_initializer='glorot_uniform')(stack(navigation_out))

    # intermediate input for control and value networks
    dynamics_in = concatenate([image_out, road_out, vehicle_out, navigation_out],
                              name='dynamics_in')
    dynamics_out = linear_combination(dynamics_in, **kwargs.get('dynamics', {}), name='dynamics-linear')
    return dynamics_out


def control_branch(inputs: dict, units: int, num_layers: int, activation=utils.swish6):
    x = inputs['dynamics']

    for _ in range(num_layers):
        x = BatchNormalization()(x)
        x = Dense(units, activation=activation, bias_initializer='glorot_uniform')(x)

    return x


# TODO: deprecated
def select_branch(branches, command):
    """Branch-selection mechanism"""
    branches_out = concatenate(branches)
    branch_size = branches[0].shape[1]
    command_size = command.shape[1]

    # multiply the branches by the command mask to select one branch, i.e. [0. 0. 0. x y z 0. 0. 0.] for branch-2
    command_mask = tf.repeat(command, repeats=branch_size, axis=1)
    selected_branch = tf.multiply(branches_out, command_mask)

    # reshape the output into a NxN square matrix then sum over rows, i.e. x = 0 + x + 0
    branch_out = tf.reshape(selected_branch, shape=(-1, command_size, branch_size))
    branch_out = tf.reduce_sum(branch_out, axis=1)
    return branch_out


# -------------------------------------------------------------------------------------------------


class PolicyNetwork(tf.keras.Model):

    def __init__(self, agent, inputs: Dict[str, Input], name='PolicyNetwork', **kwargs):
        super().__init__(inputs, outputs=self.structure(inputs, agent.num_actions, **kwargs), name=name)

        self.agent = agent

    def call(self, inputs: TensorType, **kwargs):
        dist, speed, similarity = super().call(inputs, **kwargs)

        clipped_actions = self._clip_actions(dist)
        log_prob = dist.log_prob(clipped_actions)

        mean = dist.mean()
        std = dist.stddev()

        if kwargs.get('training', False):
            mean = tf.stop_gradient(mean)
            std = tf.stop_gradient(std)

        return dict(actions=dist, log_prob=log_prob, entropy=dist.entropy(), mean=mean, std=std, speed=speed,
                    similarity=similarity)

    def structure(self, inputs: Dict[str, Input], num_actions: int, **kwargs):
        return self.policy_branch(inputs, num_actions, **kwargs)

    def policy_branch(self, inputs: Dict[str, Input], num_actions, **kwargs):
        branch = control_branch(inputs, **kwargs)
        distribution = self.get_distribution_layer(branch, num_actions)

        # auxiliary outputs
        similarity = Dense(units=1, activation=tf.nn.tanh, bias_initializer='glorot_uniform',
                           name=f'pi-similarity')(branch)

        speed = Dense(units=1, activation=lambda x: 2.0 * tf.nn.sigmoid(x), bias_initializer='glorot_uniform',
                      name=f'pi-speed')(branch)

        return distribution, speed, similarity

    @staticmethod
    def get_distribution_layer(layer: Layer, num_actions) -> tfp.layers.DistributionLambda:
        # Bounded continuous 1-dimensional actions:
        # for activations choice refer to chapter 4 of http://proceedings.mlr.press/v70/chou17a/chou17a.pdf
        # make a, b > 1, so that the Beta distribution is concave and unimodal (see paper above)
        alpha = Dense(units=num_actions, activation=utils.softplus(1.0 + 1e-2), name='alpha')(layer)
        beta = Dense(units=num_actions, activation=utils.softplus(1.0 + 1e-2), name='beta')(layer)

        return tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfp.distributions.Beta(t[0], t[1]))([alpha, beta])

    @staticmethod
    def _clip_actions(actions):
        """Clips actions to prevent numerical instability when computing (log-)probabilities.
           - Use for Beta distribution only.
        """
        return tf.clip_by_value(actions, utils.EPSILON, 1.0 - utils.EPSILON)


class CARLANetwork(Network):
    """The CARLAgent network"""

    def __init__(self, agent, control_policy: dict, control_value: dict, dynamics: dict, update_dynamics=False):
        """
        :param agent: a CARLAgent instance.
        :param control_policy: dict that specifies the policy-branch of the network.
        :param control_value: dict that specifies the value-branch of the network.
        :param dynamics: dict that specifies the architectures of the shared dynamics network.
        :param update_dynamics: set to False to prevent updating the dynamics network.
        """
        super().__init__(agent)

        # Input tensors
        self.inputs = None
        self.intermediate_inputs = None

        # Dynamics model
        self.dynamics = self.dynamics_model(**dynamics)
        self.action_index = 0

        # Value model
        self.exp_scale = 6.0
        self.value = self.value_network(**control_value)
        self.last_value = tf.zeros((1, 2), dtype=tf.float32)  # (base, exp)

        # Policy model
        self.policy = PolicyNetwork(agent=self.agent, inputs=self.intermediate_inputs, **control_policy)
        self.old_policy = PolicyNetwork(agent=self.agent, inputs=self.intermediate_inputs, **control_policy)
        self.update_old_policy()

    def value_predict(self, inputs):
        return self.value(inputs, training=False)['value']

    def predict(self, inputs: TensorType):
        dynamics_inputs = self.data_for_dynamics(inputs)
        dynamics_output = self.dynamics_predict(dynamics_inputs)

        return self._predict(inputs=dynamics_output)

    @tf.function
    def _predict(self, inputs: TensorType):
        policy = self.old_policy(inputs, training=False)
        value = self.value_predict(inputs)
        self.action_index += 1

        return policy['actions'], policy['mean'], policy['std'], policy['log_prob'], value

    def data_for_dynamics(self, inputs):
        inputs = inputs.copy()
        memory = self.agent.memory

        if tf.shape(memory.actions)[0] == 0:
            inputs['action'] = tf.zeros((1, self.agent.num_actions))
        else:
            inputs['action'] = tf.expand_dims(memory.actions[-1], axis=0)

        return inputs

    @tf.function
    def dynamics_predict(self, inputs: dict):
        return self.dynamics(inputs, training=False)

    @tf.function
    def dynamics_predict_train(self, inputs: dict):
        return self.dynamics(inputs, training=True)

    def predict_last_value(self, state, is_terminal: bool, **kwargs):
        if is_terminal:
            return self.last_value

        dynamics_data = self.data_for_dynamics(state)
        dynamics_out = self.dynamics_predict(dynamics_data)

        return self.value_predict(dynamics_out)

    def dynamics_model(self, **kwargs) -> Model:
        """Implicit Dynamics Model,
           - Inputs: state/observation, action, value
        """
        self.inputs = self._get_input_layers()
        self.inputs['action'] = Input(shape=(self.agent.num_actions,), name='action')
        dynamics_out = dynamics_layers(self.inputs, time_horizon=self.agent.env.time_horizon, **kwargs)

        self.intermediate_inputs = dict(dynamics=Input(shape=dynamics_out.shape[1:], name='dynamics'),
                                        action=self.inputs['action'])

        return Model(self.inputs, outputs=dict(dynamics=dynamics_out, action=self.inputs['action']),
                     name='Dynamics-Model')

    def _get_input_layers(self, **kwargs) -> Dict[str, Input]:
        """Transforms arbitrary complex state-spaces (and, optionally, action-spaces) as input layers"""
        input_layers = dict()

        for name, shape in self.agent.state_spec.items():
            layer = Input(shape=(self.agent.env.time_horizon,) + shape, dtype=tf.float32, name=name)
            input_layers[name] = layer

        return input_layers

    def value_network(self, **kwargs) -> Model:
        exp_scale = kwargs.pop('exponent_scale', 6.0)
        components = kwargs.pop('components', 1)
        value, speed, similarity = self.value_branch(0, exp_scale, components, **kwargs)

        outputs = dict(value=value, similarity=similarity, speed=speed)
        return Model(inputs=self.intermediate_inputs, outputs=outputs, name='Value-Network')

    def value_branch(self, index: int, exp_scale: float, components: int, **kwargs):
        branch = control_branch(self.intermediate_inputs, **kwargs)
        value = self.value_head(branch, index, exponent_scale=exp_scale, components=components)

        # auxiliary outputs
        speed = Dense(units=1, activation=lambda x: 2.0 * tf.nn.sigmoid(x), bias_initializer='glorot_uniform',
                      name=f'v-speed-{index}')(branch)
        similarity = Dense(units=1, activation=tf.nn.tanh, bias_initializer='glorot_uniform',
                           name=f'v-similarity-{index}')(branch)

        return value, speed, similarity

    def value_head(self, layer: Layer, index=0, exponent_scale=6.0, **kwargs):
        assert exponent_scale > 0.0

        base = Dense(units=1, activation=tf.nn.tanh, bias_initializer='glorot_uniform',
                     name=f'v-base-{index}')(layer)
        exp = Dense(units=1, activation=lambda x: exponent_scale * tf.nn.sigmoid(x), bias_initializer='glorot_uniform',
                    name=f'v-exp-{index}')(layer)

        return concatenate([base, exp], axis=1)

    def reset(self):
        super().reset()
        self.action_index = 0

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

        print('\n==== Dynamics Model ====')
        self.dynamics.summary()

    def save_weights(self):
        self.policy.save_weights(filepath=self.agent.weights_path['policy'])
        self.value.save_weights(filepath=self.agent.weights_path['value'])
        self.dynamics.save_weights(filepath=self.agent.dynamics_path)

    def load_weights(self, full=True):
        if full:
            self.policy.load_weights(filepath=self.agent.weights_path['policy'], by_name=False)
            self.old_policy.set_weights(self.policy.get_weights())

            self.value.load_weights(filepath=self.agent.weights_path['value'], by_name=False)
            self.dynamics.load_weights(filepath=self.agent.dynamics_path, by_name=False)
        else:
            self.dynamics.load_weights(filepath=self.agent.dynamics_path, by_name=False)
