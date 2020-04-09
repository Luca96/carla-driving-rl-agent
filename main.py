import logging
import tensorflow as tf

from agents.learn import *
from agents.learn.experiments import *
from agents.specifications import Specifications as Specs

from worlds import World
from worlds.controllers import KeyboardController

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


# TODO [project-level]: add assertions!!
# TODO [project-level]: use os.path.join for directory strings!!
# TODO [project-level]: use logger and different logging levels for debug, warnings, etc.!!

def get_client(address='localhost', port=2000, timeout=2.0):
    """Connects to the simulator.
        @:returns a client object if the CARLA simulator accepts the connection.
    """
    client = carla.Client(address, port)
    client.set_timeout(timeout)
    return client


def print_object(obj, message=None, filter=None):
    if message is not None:
        print(message)

    for x in dir(obj):
        if filter is None:
            print(x)
        elif filter in str(type(getattr(obj, x))):
            print(x)


def game_loop(vehicle='vehicle.audi.*', width=800, height=600):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = get_client(timeout=2.0)
        display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)

        world = World(client.get_world(), window_size=(width, height), actor_filter=vehicle)
        controller = KeyboardController()
        # controller = BasicAgentController(vehicle=world.player, destination=world.target_position.location)

        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(30)

            if controller.parse_events(client, world, clock, training=False):
                return
            # control = controller.act()
            # world.apply_control(control)

            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:
        if world and world.recording_enabled:
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


def test_sync_event(carla_env, timesteps, episodes, load_agent=False, agent_name='carla-agent', width=800, height=600,
                    record_dir='data/recordings', vehicle='vehicle.tesla.model3', image_shape=(150, 200, 3), **kwargs):
    pygame.init()
    pygame.font.init()

    # "lucky" seeds: 1586040996, 1586041389, 1586041713, 1586114295, 1586125577, 1586125756, 1586125933

    sync_env = carla_env(image_shape=image_shape,
                         window_size=(width, height),
                         timeout=kwargs.pop('timeout', 5.0),
                         debug=True,
                         fps=kwargs.pop('max_fps', 30.0),
                         vehicle_filter=vehicle)
    print('== ENV ==')

    # agent = Specs.carla_agent(environment=sync_env, max_episode_timesteps=timesteps, **kwargs)
    agent = Agent.create(agent='tensorforce', environment=sync_env, max_episode_timesteps=timesteps, **kwargs)
    print('== AGENT ==')

    try:
        if load_agent:
            agent.load(directory='weights/agents', filename=agent_name, environment=sync_env)
            print('Agent loaded.')

        sync_env.learn(agent, num_episodes=episodes, max_episode_timesteps=timesteps, agent_name=agent_name,
                       record_dir=record_dir)

    finally:
        sync_env.close()
        pygame.quit()


def test_curriculum_learning(timesteps, episodes, load_agent=False, agent_name='carla-agent',
                             vehicle='vehicle.tesla.model3', width=800, height=600, image_shape=(150, 200, 3),
                             **kwargs):
    pygame.init()
    pygame.font.init()

    env = CurriculumCARLAEnvironment(image_shape=image_shape,
                                     window_size=(width, height),
                                     timeout=kwargs.pop('timeout', 5.0),
                                     debug=True,
                                     fps=kwargs.pop('max_fps', 30.0),  # TODO: crash with 60 fps !??
                                     vehicle_filter=vehicle)
    print('== ENV ==')

    agent = Specs.carla_agent(environment=env, max_episode_timesteps=timesteps, **kwargs)
    print('== AGENT ==')

    try:
        env.learn(agent, initial_timesteps=64, max_timesteps=timesteps, difficulty=3, num_stages=episodes,
                  load_agent=load_agent, agent_name=agent_name, increment=5, max_repetitions=3)
        print('Curriculum Learning completed.')
    finally:
        print('Closing...')
        env.close()
        pygame.quit()


def decay(value: float, rate: float, steps: int):
    for _ in range(steps):
        value *= rate

    return value


if __name__ == '__main__':
    # TODO: to make Tensorforce work with tensorflow 2.0.1, comment line 29 and 30 in
    #  '/usr/lib/python3.8/site-packages/tensorflow_core/lite/experimental/microfrontend/python/ops
    #  /audio_microfrontend_op.py'

    # ./CarlaUE4.sh -windowed -ResX=8 -ResY=8 -benchmark -fps=30
    # https://docs.unrealengine.com/en-US/Programming/Basics/CommandLineArguments/index.html

    # game_loop()

    num_episodes = 3
    batch_size = 256
    frequency = batch_size
    num_timesteps = batch_size * 3

    # TODO: provide a base CARLA env class for a sync, async, pretrain environments..
    # TODO: do a common CARLA world wrapper for the environments. (or a world_utils module)

    tfargs = dict(policy=dict(network=Specs.agent_network(conv=dict(stride=1, pooling='max', filters=36),
                                                          final=dict(layers=2, units=256, activation='leaky-relu')),
                              optimizer=dict(type='evolutionary',
                                             num_samples=6,
                                             learning_rate=Specs.exp_decay(steps=num_timesteps,
                                                                           unit='updates',
                                                                           initial_value=0.1,
                                                                           rate=0.995)),
                              temperature=0.90),

                  batch_size=batch_size,
                  update_frequency=frequency,

                  critic=dict(network=Specs.agent_network(conv=dict(stride=1, pooling='max', filters=36),
                                                          final=dict(layers=2, units=160)),
                              optimizer=dict(type='adam', learning_rate=3e-3),
                              temperature=0.70),

                  discount=1.0,
                  horizon=100,

                  preprocessing=dict(image=[dict(type='image', width=140, height=105, grayscale=True),
                                            dict(type='exponential_normalization')]),

                  summarizer=Specs.summarizer(frequency=frequency),

                  entropy_regularization=Specs.exp_decay(steps=num_timesteps, initial_value=0.1, rate=0.995),
                  exploration=0.0 + 0.1)

    tfargs = Specs.agent_v1(batch_size, frequency, num_timesteps, filters=36,
                            lr=decay(value=decay(0.1, 0.995, steps=180), rate=0.995, steps=180),
                            temperature=(decay(0.9, 0.995, 180), decay(0.7, 0.995, 180)))

    # test_sync_event(CARLAExperiment3, num_timesteps, num_episodes * 20, width=670, height=500,
    #                 load_agent=True, agent_name='carla-agent-evo-f40', record_dir=None, **tfargs)

    experiment = CARLABaselineExperiment(window_size=(670, 500), debug=True)
    experiment.run(agent_args=dict(max_episode_timesteps=512, batch_size=512),
                   num_episodes=10, max_episode_timesteps=512, record_dir=None, weights_dir=None)
