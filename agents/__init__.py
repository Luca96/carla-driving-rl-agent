
from tensorforce import Agent


class AgentConfigs(object):
    """A class with pre-defined agents."""
    POLICY_GRADIENT = dict(type='policy_gradient', clipping_value=0.2)
    RECENT_MEMORY = dict(type='recent')

    @staticmethod
    def ppo(env, timesteps, **kwargs):
        return Agent.create(agent='ppo',
                            environment=env,
                            max_episode_timesteps=timesteps)

    @staticmethod
    def random(env, timesteps, **kwargs):
        return Agent.create(agent='random',
                            environment=env,
                            max_episode_timesteps=timesteps)

    @staticmethod
    def tensorforce(env, timesteps, batch_size, horizon, optimizer='adam', discount=0.95, **kwargs):
        return Agent.create(agent='tensorforce',
                            environment=env,
                            max_episode_timesteps=timesteps,
                            memory=AgentConfigs.RECENT_MEMORY,
                            objective=AgentConfigs.POLICY_GRADIENT,
                            update=dict(unit='timesteps',
                                        batch_size=batch_size),
                            optimizer=optimizer,
                            reward_estimation=dict(horizon=horizon,
                                                   discount=discount,
                                                   # estimate_horizon='early',
                                                   # estimate_actions=True,
                                                   # estimate_advantage=True
                                                   ),
                            **kwargs)

    @staticmethod
    def tensorforce2(env, **kwargs):
        return Agent.create(agent='tensorforce',
                            environment=env,
                            **kwargs)

    @staticmethod
    def load(config_name, **kwargs):
        pass
