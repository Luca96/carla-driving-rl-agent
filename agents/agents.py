from tensorforce import Agent

from agents.specifications import Specifications as Specs
from agents.learn.environment import SynchronousCARLAEnvironment


class Agents:
    """Provides predefined agents"""

    @staticmethod
    def baseline(carla_env: SynchronousCARLAEnvironment, max_episode_timesteps=512, batch_size=256, update_frequency=64,
                 horizon=200, discount=0.997, exploration=0.0, entropy=0.05, **kwargs) -> Agent:
        return Agent.create(agent='tensorforce',
                            environment=carla_env,
                            max_episode_timesteps=max_episode_timesteps,

                            policy=Specs.policy(distributions='gaussian',
                                                network=Specs.agent_network_v0(),
                                                temperature=0.99),

                            optimizer=dict(type='adam', learning_rate=3e-4),
                            objective=Specs.obj.policy_gradient(clipping_value=0.2),
                            update=Specs.update(unit='timesteps', batch_size=batch_size, frequency=update_frequency),

                            reward_estimation=dict(horizon=horizon,
                                                   discount=discount,
                                                   estimate_advantage=True),
                            exploration=exploration,
                            entropy_regularization=entropy,
                            **kwargs)

    @staticmethod
    def evolutionary(carla_env: SynchronousCARLAEnvironment, max_episode_timesteps: int, batch_size=256,
                     update_frequency=256, decay_steps=768, filters=36, decay=0.995, lr=0.1, units=(256, 128),
                     layers=(2, 2), temperature=(0.9, 0.7), horizon=100, width=140, height=105, **kwargs) -> Agent:

        policy_net = Specs.agent_network(conv=dict(stride=1, pooling='max', filters=filters),
                                         final=dict(layers=layers[0], units=units[0], activation='leaky-relu'))

        decay_lr = Specs.exp_decay(steps=decay_steps, unit='updates', initial_value=lr, rate=decay)

        critic_net = Specs.agent_network(conv=dict(stride=1, pooling='max', filters=filters),
                                         final=dict(layers=layers[1], units=units[1]))

        return Specs.carla_agent(carla_env, max_episode_timesteps,
                                 policy=dict(network=policy_net,
                                             optimizer=dict(type='evolutionary', num_samples=6, learning_rate=decay_lr),
                                             temperature=temperature[0]),

                                 batch_size=batch_size,
                                 update_frequency=update_frequency,

                                 critic=dict(network=critic_net,
                                             optimizer=dict(type='adam', learning_rate=3e-3),
                                             temperature=temperature[1]),
                                 discount=1.0,
                                 horizon=horizon,

                                 preprocessing=dict(image=[dict(type='image', width=width, height=height, grayscale=True),
                                                           dict(type='exponential_normalization')]),

                                 summarizer=Specs.summarizer(frequency=update_frequency),

                                 entropy_regularization=Specs.exp_decay(steps=decay_steps, unit='updates',
                                                                        initial_value=lr, rate=decay),
                                 **kwargs)
