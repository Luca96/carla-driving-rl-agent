"""A collection of "stages" for curriculum learning"""

import carla

from typing import Callable, List, Optional
from tensorforce.agents import TensorforceAgent

from agents import env_utils


class Stage(object):
    """Base class for stages (aka challenges)"""

    def __init__(self, agent: TensorforceAgent, env_class, learn_trials: int, target_episodic_reward: float,
                 success_rate: float, eval_trials=1, pretrain: Optional[dict] = None, **env_args):
        assert learn_trials > 0
        assert eval_trials > 0
        assert 0.0 < success_rate <= 1.0
        assert isinstance(agent, TensorforceAgent)

        self.agent = agent
        self.environment = None
        self.env_class = env_class
        self.env_args = env_args if isinstance(env_args, dict) else dict()

        # a stage is successful when the agent reaches an episodic reward greater or equal to target_episodic_reward a
        # number of times (fraction - percentage) greater or equal to target_rate
        self.success_rate = success_rate
        self.target_reward = target_episodic_reward

        self.learn_trials = learn_trials
        self.eval_trials = eval_trials
        self.episodic_rewards = []  # related to evaluation

        if isinstance(pretrain, dict):
            self.should_pretrain = True
            self.traces_dir = pretrain.get('traces', 'data/traces')
            self.num_traces = pretrain.get('num_iterations')

    @staticmethod
    def create(stage_spec: dict):
        pass

    def learn(self):
        self.environment.learn2(agent=self.agent, num_episodes=self.learn_trials)

    def evaluate(self):
        self.episodic_rewards = self.environment.evaluate(agent=self.agent, num_episodes=self.eval_trials)

    def run(self) -> bool:
        self.pretrain()

        self.setup()
        self.learn()
        self.evaluate()

        self.cleanup()
        return self.is_successful()

    def pretrain(self):
        if self.should_pretrain:
            self.agent.pretrain(directory=self.traces_dir, num_iterations=self.num_traces)

    def setup(self):
        self.environment = self.env_class(**self.env_args)

    def cleanup(self):
        self.episodic_rewards.clear()
        self.environment.close()

    def is_successful(self) -> bool:
        count = 0

        for episode_reward in self.episodic_rewards:
            if episode_reward >= self.target_reward:
                count += 1

        return self.eval_trials / count >= self.success_rate


class FixedOriginStage(Stage):
    def __init__(self, spawn_point: carla.Transform, *args, **kwargs):
        super().__init__(*args, env_args=dict(spawn_point=spawn_point, **kwargs))


# -------------------------------------------------------------------------------------------------
# -- Curriculum Learning
# -------------------------------------------------------------------------------------------------

class CurriculumLearning(object):

    def __init__(self, agent_spec: dict, env_spec: dict, curriculum: List[dict], save: dict = None):
        """
        :param save: dict(directory, frequency, filename, append)
        :param agent_spec: dict(callable=,, **kwargs)
        :param env_spec: dict(callable=, **kwargs)
        :param curriculum: [dict(learn_episodes, eval_episodes, environment=dict(), pretrain=dict(...)), ...]
        """
        assert isinstance(agent_spec, dict)
        assert isinstance(env_spec, dict)
        assert isinstance(curriculum, list) and len(curriculum) > 0

        self.agent_class = agent_spec.pop('callable')
        self.agent_args = agent_spec
        self.env_class = env_spec.pop('callable')
        self.env_args = env_spec
        self.save = save
        self.curriculum = curriculum

        # TODO: check curriculum specs before start()!

    def pretrain(self, agent, spec: dict):
        if spec is None:
            return

        env_utils.pretrain(agent, num_traces=spec['num_traces'], traces_dir=spec['traces_dir'], save=self.save)

    @staticmethod
    def is_successful(stage: dict, episodic_rewards: List[float]) -> (bool, float, float):
        count = 0

        for episode_reward in episodic_rewards:
            if episode_reward >= stage['target_reward']:
                count += 1

        # some statistics
        avg_reward = sum(episodic_rewards) / len(episodic_rewards)
        rate = len(episodic_rewards) / count
        success = rate >= stage['success_rate']
        return success, rate, avg_reward

    def start(self):
        for i, stage in enumerate(self.curriculum):
            print(f"Stage-{i}")
            agent, environment = self.initialize(**stage.get('environment', dict()))

            self.pretrain(agent, spec=stage.get('pretrain', None))

            for j in range(stage.get('repeat', 1)):
                print(f"\tTrial-{i}")
                environment.learn3(agent, num_episodes=stage['learn_episodes'], save=self.save)
                episodic_rewards = environment.evaluate(agent, num_episodes=stage['eval_episodes'])

                success, success_rate, average_reward = self.is_successful(stage, episodic_rewards)

                if success:
                    print(f"\t\tSuccess! Average reward {round(average_reward, 2)}")
                    break
                else:
                    print(f"\t\tFailure!, Avg. Reward {round(average_reward, 2)}, rate {round(success_rate, 2)}")

    def initialize(self, **kwargs):
        # TODO: could be useful to overwrite base env arguments
        environment = self.env_class(**self.env_args, **kwargs)

        agent = env_utils.load_agent(directory=self.save['directory'], filename=self.save['filename'],
                                     environment=environment, from_function=self.agent_class, **self.agent_args)
        return agent, environment
