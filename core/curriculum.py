"""Curriculum Learning for CARLA Agent"""


# -------------------------------------------------------------------------------------------------
# -- Stages
# -------------------------------------------------------------------------------------------------

class Stage(object):
    """Base class for stages (aka challenges)"""

    # TODO: class assertions
    def __init__(self, agent: dict, environment: dict, imitation: dict = None):
        assert isinstance(agent, dict)
        assert isinstance(environment, dict)

        # Agent
        self.agent_class = agent.get('class', agent['class_'])
        self.agent_args = agent.get('args', {})
        self.agent_learn_args = agent.get('learn', {})
        self.agent = None

        # Environment
        self.env_class = environment.get('class', environment['class_'])
        self.env_args = environment.get('args', {})
        self.env = None

        # Imitation learning
        if isinstance(imitation, dict):
            self.should_imitate = True
            self.imitation_class = imitation.get('class', imitation['class_'])
            self.imitation_args = imitation.get('args', {})
            self.imitation_learn_args = imitation.get('learn', {})
        else:
            self.should_imitate = False
        self.imitation_agent = None

    def init(self):
        # init env
        self.env = self.env_class(**self.env_args)

        # init agent
        self.agent = self.agent_class(self.env, **self.agent_args)

    def learn(self):
        self.agent.learn(**self.agent_learn_args)

    def imitate(self):
        # init imitation agent
        self.imitation_agent = self.imitation_class(self.agent, **self.imitation_args)
        self.imitation_agent.imitate(**self.imitation_learn_args)

    def run(self):
        self.init()

        if self.should_imitate:
            self.imitate()

        input('press enter to continue...')

        self.learn()
        self.cleanup()

    def cleanup(self):
        self.env = None
        self.agent = None
        self.imitation_agent = None
