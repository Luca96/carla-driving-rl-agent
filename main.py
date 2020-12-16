import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import pygame
import logging
import tensorflow as tf

tf.test.is_gpu_available()

from core import learning

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


def collect_experience(timesteps=512, threshold=0.75, amount=25, name='imitation', behaviour='normal', town=None):
    """Collects experience traces (npz file format) by running the CARLA's privileged agent."""
    # 'normal', 'cautious', or 'aggressive'
    from core import CARLAEnv
    args = dict(timesteps=timesteps, threshold=timesteps * threshold, image_shape=(90, 120, 3), window_size=(720, 180),
                render=True, debug=False, env_class=CARLAEnv, town=town, name=name, behaviour=behaviour)

    learning.collect_experience(episodes=amount, **args)
    learning.collect_experience(episodes=amount * 2, spawn=dict(vehicles=20, pedestrians=50), **args)
    learning.collect_experience(episodes=amount * 2, spawn=dict(vehicles=50, pedestrians=150), **args)


if __name__ == '__main__':
    # # ---- COLLECT EXPERIENCE (~300 traces)
    # collect_experience(amount=40, threshold=0.75, name='imitation', behaviour='normal')
    # collect_experience(amount=20, threshold=0.75, name='imitation', behaviour='aggressive')

    # # -- IMITATION LEARNING (5 epochs)
    # learning.imitation_learning(batch_size=64, lr=3e-4, seed=42, epochs=5,
    #                             alpha=1.0, beta=1.0, clip=0.5, name='imitation-final')

    # TODO: ""Evaluation""
    #  - evaluate the model on different horizons: 64, 128, 256, 512, 768, 1024
    #  - set the time budget to 5km/h (or NO time budget at all!)
    #  - evaluate on town: 1, 2, 3, 7?, 10? (all available towns?)
    #  - use "random agent" as baseline

    # CURRICULUM LEARNING:
    # -- STAGE-1 --
    # learning.stage_s1(episodes=5, timesteps=512, batch_size=64, gamma=0.999, lambda_=0.999, save_every='end',
    #                   update_frequency=1, policy_lr=3e-4, value_lr=3e-4, dynamics_lr=3e-4,
    #                   clip_ratio=0.2, entropy_regularization=1.0, load=False, seed_regularization=True,
    #                   seed=51, polyak=0.999, aug_intensity=0.0, repeat_action=1, load_full=False)\
    #     .run2(epochs=200, epoch_offset=0)
    # exit()

    # -- STAGE-2 --
    # learning.stage_s2(episodes=5, timesteps=512, batch_size=64, gamma=0.9999, lambda_=0.999, save_every='end',
    #                   update_frequency=1, policy_lr=2e-4, value_lr=3e-4, dynamics_lr=3e-4,
    #                   optimization_steps=(3, 1),
    #                   clip_ratio=0.15, entropy_regularization=1.0, seed_regularization=True,
    #                   seed=51, polyak=1.0, aug_intensity=0.0, repeat_action=1) \
    #     .run2(epochs=100, epoch_offset=0)
    #
    # stage2 = learning.stage_s2(episodes=5, timesteps=512, batch_size=64, gamma=0.9999, lambda_=0.999, save_every='end',
    #                            update_frequency=1, policy_lr=3e-5, value_lr=3e-5, dynamics_lr=3e-3,
    #                            optimization_steps=(2, 1),
    #                            clip_ratio=0.15, entropy_regularization=2.0, seed_regularization=True,
    #                            seed=51, polyak=1.0, aug_intensity=0.0, repeat_action=1)
    #
    # # stage2.run2(epochs=50, epoch_offset=100)
    # stage2.evaluate(name='eval-512-100-town3-clearNoon', timesteps=512, trials=100, seeds='sample')
    # exit()

    # -- STAGE-3 --
    # stage3 = learning.stage_s3(episodes=5, timesteps=512, batch_size=64, gamma=0.9999, lambda_=0.999, save_every='end',
    #                            update_frequency=1, policy_lr=3e-5, value_lr=3e-5, dynamics_lr=3e-4,
    #                            clip_ratio=0.125, entropy_regularization=1.0, seed_regularization=True,
    #                            seed=51, polyak=1.0, aug_intensity=0.0, repeat_action=1)
    #
    # stage3.run2(epochs=100, epoch_offset=0)
    # stage3.evaluate(name='eval-512-100-town3-lightWeather', timesteps=512, trials=100, seeds='sample')
    # exit()

    # -- STAGE-4 --
    stage4 = learning.stage_s4(episodes=5, timesteps=512, batch_size=64, gamma=0.9999, lambda_=0.999, save_every='end',
                               update_frequency=1, policy_lr=3e-5, value_lr=3e-5, dynamics_lr=3e-4,
                               clip_ratio=0.125, entropy_regularization=1.0, seed_regularization=True,
                               seed=51, polyak=1.0, aug_intensity=0.0, repeat_action=1, town='Town01')

    # stage4.run2(epochs=50, epoch_offset=0)
    stage4.evaluate(name='eval-512-50-town1-lightWeather', timesteps=512, trials=5, seeds='sample')
    exit()

    pygame.quit()
