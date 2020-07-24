import os
import pygame
import logging
import tensorflow as tf

import tests


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


if __name__ == '__main__':
    # tests.curriculum_learning()
    pygame.quit()
