"""Test cases"""

from agents.experiments import *
from agents.environment import SynchronousCARLAEnvironment


def test_carla_env():
    env = SynchronousCARLAEnvironment(debug=True)

    env.train(agent=Agents.evolutionary(env, max_episode_timesteps=200, update_frequency=64, batch_size=64, horizon=32),
              num_episodes=5, max_episode_timesteps=200, weights_dir=None, record_dir=None)


def test_baseline_env():
    env = CARLABaselineExperiment(debug=True)
    env.train(agent=None, num_episodes=5, max_episode_timesteps=200, weights_dir=None, record_dir=None)


def test_pretrain_env(num_episodes=5):
    # Pretraining:
    for i in range(1, num_episodes + 1):
        print(f'Trace-{i}')
        experiment = CARLAPretrainExperiment(window_size=(670, 500), debug=True, vehicle_filter='vehicle.tesla.model3')
        experiment.train(agent=Agents.dummy.random_walk(experiment, max_episode_timesteps=256, speed=30,
                                                        traces_dir='data/traces/pretrain_tesla_batch256'),
                         num_episodes=1, max_episode_timesteps=256, record_dir=None, weights_dir=None, load_agent=False)


def test_sequence_layer():
    env = CARLARouteFollowExperiment(debug=True)
    agent = Agents.criticless(env, max_episode_timesteps=768, filters=40,
                              preprocessing=dict(image=[dict(type='image', width=140, height=105, grayscale=True),
                                                        # dict(type='instance_normalization'),
                                                        dict(type='sequence', length=20)]))

    env.train(agent, num_episodes=5, max_episode_timesteps=768, weights_dir=None, record_dir=None)


def test_keyboard_agent():
    env = CARLAPlayEnvironment(debug=True, image_shape=(600, 800, 3), vehicle_filter='vehicle.audi.*')
    env.train(None, num_episodes=10, max_episode_timesteps=1024, weights_dir=None, record_dir=None)


def test_route_follow_segmentation(num_episodes: int, num_timesteps: int):
    env = CARLARouteFollowExperiment(vehicle_filter='vehicle.tesla.model3', debug=True)
    # env = CARLASegmentationExperiment(vehicle_filter='vehicle.tesla.model3', debug=True)
    env.train(agent=None, num_episodes=num_episodes, max_episode_timesteps=num_timesteps,
              agent_name='carla-segmentation-experiment', load_agent=False, record_dir=None, weights_dir=None)

# TODO: pretrain agent, then train
# TODO: try RNN
# TODO: adam vs evolutionary optimizer
# TODO: 1 vs N past inputs in observation space
