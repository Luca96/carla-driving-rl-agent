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

    # -- IMITATION LEARNING (5 epochs)
    learning.imitation_learning(batch_size=64, lr=3e-4, seed=42, epochs=5,
                                alpha=1.0, beta=1.0, clip=0.5, name='imitation-final')
    exit()

    # CURRICULUM LEARNING:
    # -- STAGE-1 --
    learning.stage_s1(episodes=5, timesteps=64 * 3, gamma=0.999, lambda_=0.995, save_every='end',
                      seed=42, polyak=0.999, aug_intensity=1.0, repeat_action=5, load_full=False)\
        .run2(epochs=10, epoch_offset=0)

    # learning.stage_s1(episodes=5 * 8, timesteps=512*2, batch_size=256, gamma=0.999, lambda_=0.995, save_every='end',
    #                   update_frequency=8, clip_ratio=0.1, policy_lr=3e-4, value_lr=3e-4,
    #                   entropy_regularization=0.1,
    #                   seed=42, polyak=0.999, aug_intensity=0.0, repeat_action=1, load_full=True) \
    #     .run2(epochs=10, epoch_offset=40)

    # -- STAGE-2 --
    learning.stage_s2(episodes=5, timesteps=64 * 3, gamma=0.999, lambda_=0.995, save_every='end',
                      seed=42, polyak=0.999, aug_intensity=1.0, repeat_action=4, load_full=True) \
        .run2(epochs=10, epoch_offset=0)

    # learning.stage_s2(episodes=5 * 16, timesteps=512, batch_size=256, gamma=0.999, lambda_=0.995, save_every='end',
    #                   update_frequency=16, clip_ratio=0.1, policy_lr=3e-4, value_lr=3e-4,
    #                   entropy_regularization=0.1,
    #                   seed=42, polyak=0.999, aug_intensity=0.0, repeat_action=1, load_full=True) \
    #     .run2(epochs=30, epoch_offset=0)
    # exit()

    # -- STAGE-3 --
    learning.stage_s3(episodes=5, timesteps=64 * 3, gamma=0.999, lambda_=0.995, save_every='end',
                      seed=42, polyak=0.999, aug_intensity=1.0, repeat_action=4, load_full=True) \
        .run2(epochs=20, epoch_offset=0)

    # -- STAGE-4 --
    learning.stage_s4(episodes=5, timesteps=64 * 3, town='Town01', gamma=0.999, lambda_=0.995, save_every='end',
                      seed=42, polyak=0.999, aug_intensity=1.0, repeat_action=4, load_full=True) \
        .run2(epochs=5, epoch_offset=0)

    learning.stage_s4(episodes=5, timesteps=64 * 3, town='Town02', gamma=0.999, lambda_=0.995, save_every='end',
                      seed=42, polyak=0.999, aug_intensity=1.0, repeat_action=4, load_full=True) \
        .run2(epochs=5, epoch_offset=0)

    pygame.quit()
