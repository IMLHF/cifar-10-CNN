import tensorflow as tf


class basic_config(object):
  learning_rate = 0.001
  CIFAR_10_DIR = 'cifar-10-batches-py'
  BATCH_SIZE = 128
  GPU_RAM_ALLOW_GROWTH = True
  EPOCHS = 50
  SAVE_DIR = 'exp'
  CHECK_POINT = 'nnet'


class C001(basic_config):
  ACTIVATION = tf.nn.tanh
  CONV_PADDING = 'VALID'
  POOL_PADDING = 'VALID'
  DROP_RATE = 0.2


PARAM = C001
