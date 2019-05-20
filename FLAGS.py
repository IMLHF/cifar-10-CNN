import tensorflow as tf

class basic_config(object):
    learning_rate = 0.001
    CIFAR_10_DIR = 'cifar-10-batches-py'

class config1(basic_config):
    ACTIVATION = tf.nn.tanh
    CONV_PADDING='VALID'
    POOL_PADDING='VALID'
    DROP_RATE = 0.2

PARAM = config1
