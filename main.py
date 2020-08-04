import os
import pygame
import logging
import tensorflow as tf

from core import learning


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


if __name__ == '__main__':
    learning.stage_1(episodes=10 - 9, timesteps=512 // 4, seed=123).run(repeat=5, collect=0)

    pygame.quit()
