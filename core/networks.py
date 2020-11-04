"""Networks architectures for CARLA Agent"""

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from typing import List, Dict, Union

from rl import networks
from rl import utils


# -------------------------------------------------------------------------------------------------
# -- FEATURE-NET (feed-forward nn for feature vector)
# -------------------------------------------------------------------------------------------------

def feature(input_feature: Layer, units=32, num_layers=2, activation=utils.swish6, normalization='batch'):
    if normalization == 'batch':
        x = BatchNormalization()(input_feature)
    else:
        x = input_feature

    for _ in range(num_layers):
        x = Dense(units, activation=activation)(x)

    return x


# -------------------------------------------------------------------------------------------------
# -- SHARED NETWORK (P_\psi)
# -------------------------------------------------------------------------------------------------

def dynamics_layers(inputs: dict, **kwargs):
    """Defined the shared-network architecture, returns its last layer"""
    # process observations:
    image_out = networks.shufflenet_v2(inputs['state_image'], **kwargs.get('shufflenet', {}))
    image_out = feature(image_out, **kwargs.get('image', dict(noise=0.0, units=128)))

    road_out = feature(inputs['state_road'], **kwargs.get('road', dict(normalization=None)))
    vehicle_out = feature(inputs['state_vehicle'], **kwargs.get('vehicle', {}), )
    control_out = feature(inputs['state_past_control'], **kwargs.get('control', {}), )
    command_out = feature(inputs['state_command'], **kwargs.get('command', {}), )

    agent_in = concatenate([inputs['action'], inputs['value'], inputs['reward']])
    agent_out = feature(agent_in, **kwargs.get('agent', {}))

    dynamics_out = concatenate([command_out, control_out, image_out, road_out, vehicle_out, agent_out],
                               name='dynamics_out')
    return dynamics_out, []


def control_branch(inputs: dict, units: int, num_layers: int, activation=utils.swish6):
    x = inputs['dynamics']

    for _ in range(num_layers):
        x = BatchNormalization()(x)
        x = Dense(units, activation=activation)(x)

    return x


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

class CARLANetwork(networks.PPONetwork):
    """The CARLAgent network"""

    def __init__(self, agent, control_policy: dict, control_value: dict, dynamics: dict, update_dynamics=False):
        """
        :param agent: a CARLAgent instance.
        :param control_policy: dict that specifies the policy-branch of the network.
        :param control_value: dict that specifies the value-branch of the network.
        :param dynamics: dict that specifies the architectures of the shared dynamics network.
        :param update_dynamics: set to False to prevent updating the dynamics network.
        """
        self.agent = agent  # workaround
        self.recurrences = []
        self.should_update_dynamics = update_dynamics

        # Input tensors
        self.inputs = None
        self.intermediate_inputs = None

        # Dynamics model
        self.dynamics = self.dynamics_model(**dynamics)

        # Imitation pretraining model: dynamics + policy & value
        self.imitation = None
        self.inference = None

        self.beta = None
        self.action_index = 0

        super().__init__(agent, policy=control_policy, value=control_value)

    @tf.function
    def act(self, inputs: Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]]):
        dynamics_out = self.dynamics_predict(inputs)
        action = self.predict_actions(dynamics_out)
        value = self.value_predict(dynamics_out)
        return [action, value]

    @tf.function
    def predict_actions(self, inputs):
        return self.policy(inputs, training=False)['actions']

    def value_predict(self, inputs):
        return super().value_predict(inputs)['value']

    def predict(self, inputs: Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]]):
        dynamics_inputs = self.data_for_dynamics(inputs)
        dynamics_output = self.dynamics_predict(dynamics_inputs)

        return self._predict(inputs=dynamics_output)

    @tf.function
    def _predict(self, inputs: Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]]):
        policy = self.old_policy(inputs, training=False)
        value = self.value_predict(inputs)
        self.action_index += 1

        return policy['actions'], policy['mean'], policy['std'], policy['log_prob'], value

    def data_for_dynamics(self, inputs):
        inputs = inputs.copy()
        memory = self.agent.memory

        if tf.shape(memory.actions)[0] == 0:
            inputs['value'] = tf.zeros_like(self.last_value)
            inputs['reward'] = tf.zeros((1, 1))
            inputs['action'] = tf.zeros((1, self.agent.num_actions))
        else:
            inputs['value'] = tf.expand_dims(memory.values[-1], axis=0)
            inputs['action'] = tf.expand_dims(memory.actions[-1], axis=0)
            inputs['reward'] = tf.expand_dims(memory.rewards[-1], axis=0)

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

    def policy_layers(self, inputs: Dict[str, Input], **kwargs) -> Layer:
        return control_branch(inputs, **kwargs)

    def dynamics_model(self, **kwargs) -> Model:
        """Implicit Dynamics Model,
           - Inputs: state/observation, action, value
        """
        self.inputs = self._get_input_layers()
        self.inputs['value'] = Input(shape=(2,), name='value')
        self.inputs['reward'] = Input(shape=(1,), name='reward')
        self.inputs['action'] = Input(shape=(self.agent.num_actions,), name='action')

        dynamics_out, self.recurrences = dynamics_layers(self.inputs, **kwargs)

        self.intermediate_inputs = dict(dynamics=Input(shape=dynamics_out.shape[1:], name='dynamics'),
                                        command=self.inputs['state_command'], action=self.inputs['action'])

        return Model(self.inputs, outputs=dict(dynamics=dynamics_out, command=self.inputs['state_command'],
                                               action=self.inputs['action']), name='Dynamics-Model')

    def policy_network(self, **kwargs) -> Model:
        num_actions = self.agent.num_actions
        # command = self.intermediate_inputs['command']

        # branches = [self.policy_branch(i, **kwargs) for i in range(command.shape[1])]
        # branch_out = select_branch(branches, command)
        branch_out = self.policy_branch(index=0, **kwargs)

        outputs = dict(actions=branch_out[:, 0:num_actions], log_prob=branch_out[:, num_actions:num_actions * 2],
                       old_log_prob=branch_out[:, num_actions * 2:num_actions * 3], mean=branch_out[:, -5],
                       std=branch_out[:, -4], entropy=branch_out[:, -3],
                       similarity=branch_out[:, -2], speed=branch_out[:, -1])
        return Model(inputs=self.intermediate_inputs, outputs=outputs, name='Policy-Network')

    def policy_branch(self, index: int, **kwargs):
        branch = self.policy_layers(self.intermediate_inputs, **kwargs)
        distribution_out = self.get_distribution_layer(branch, index)

        # auxiliary outputs
        similarity = Dense(units=1, activation=tf.nn.tanh, name=f'pi-similarity-{index}')(branch)
        speed = Dense(units=1, activation=lambda x: 2.0 * tf.nn.sigmoid(x), name=f'pi-speed-{index}')(branch)

        return concatenate([*distribution_out, similarity, speed])

    def value_network(self, **kwargs) -> Model:
        exp_scale = kwargs.pop('exponent_scale', 6.0)
        components = kwargs.pop('components', 1)
        # command = self.intermediate_inputs['command']

        # branches = [self.value_branch(i, exp_scale, components, **kwargs) for i in range(command.shape[1])]
        # branch_out = select_branch(branches, command)
        branch_out = self.value_branch(0, exp_scale, components, **kwargs)

        outputs = dict(value=branch_out[:, 0:2], similarity=branch_out[:, 2], speed=branch_out[:, 3])
        return Model(inputs=self.intermediate_inputs, outputs=outputs, name='Value-Network')

    def value_branch(self, index: int, exp_scale: float, components: int, **kwargs):
        branch = self.value_layers(self.intermediate_inputs, **kwargs)
        value = self.value_head(branch, index, exponent_scale=exp_scale, components=components)

        # auxiliary outputs
        speed = Dense(units=1, activation=lambda x: 2.0 * tf.nn.sigmoid(x), name=f'v-speed-{index}')(branch)
        similarity = Dense(units=1, activation=tf.nn.tanh, name=f'v-similarity-{index}')(branch)

        return concatenate([value, similarity, speed])

    def value_head(self, layer: Layer, index=0, exponent_scale=6.0, **kwargs):
        assert exponent_scale > 0.0

        base = Dense(units=1, activation=tf.nn.tanh, name=f'v-base-{index}')(layer)
        exp = Dense(units=1, activation=lambda x: exponent_scale * tf.nn.sigmoid(x), name=f'v-exp-{index}')(layer)

        return concatenate([base, exp], axis=1)

    def get_distribution_layer(self, layer: Layer, index=0) -> tfp.layers.DistributionLambda:
        num_actions = self.agent.num_actions

        alpha = Dense(units=num_actions, activation=utils.softplus(1.0 + 1e-2), name=f'alpha-{index}')(layer)
        beta = Dense(units=num_actions, activation=utils.softplus(1.0 + 1e-2), name=f'beta-{index}')(layer)

        self.beta = tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Beta(t[0], t[1]),
                convert_to_tensor_fn=self._distribution_to_tensor)([alpha, beta])

        return self.beta

    def _distribution_to_tensor(self, d: tfp.distributions.Distribution):
        actions = d.sample()
        old_actions = self.intermediate_inputs['action'][self.action_index]

        log_prob = d.log_prob(self._clip_actions(actions))
        old_log_prob = d.log_prob(self._clip_actions(old_actions))

        return actions, log_prob, old_log_prob, d.mean(), d.stddev(), d.entropy()

    @staticmethod
    def _clip_actions(actions):
        return tf.clip_by_value(actions, utils.EPSILON, 1.0 - utils.EPSILON)

    def imitation_model(self):
        if self.imitation is not None:
            return

        # link dynamics with policy and value networks
        dynamics_out = self.dynamics(self.inputs)
        policy_out = self.policy(dynamics_out)
        value_out = self.value(dynamics_out)

        self.imitation = Model(self.inputs, outputs=[policy_out, value_out], name='Imitation-Model')

    @tf.function
    def imitation_predict(self, inputs: dict):
        return self.imitation(inputs, training=True)

    def init_inference_model(self):
        if self.inference is not None:
            return

        dynamics_out = self.dynamics(self.inputs)
        policy_out = self.policy(dynamics_out)
        value_out = self.value(dynamics_out)

        self.inference = Model(self.inputs, outputs=[policy_out, value_out], name='Inference-Model')

    def reset(self):
        super().reset()
        self.action_index = 0

        for gru in self.recurrences:
            gru.reset_states()

    def summary(self):
        super().summary()
        print('\n==== Dynamics Model ====')
        self.dynamics.summary()

    def save_weights(self):
        super().save_weights()
        self.dynamics.save_weights(filepath=self.agent.dynamics_path)

    def load_weights(self, full=True):
        if full:
            super().load_weights()
            self.dynamics.load_weights(filepath=self.agent.dynamics_path, by_name=False)
        else:
            self.dynamics.load_weights(filepath=self.agent.dynamics_path, by_name=False)
