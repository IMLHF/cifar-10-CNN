import tensorflow as tf


class base_config(object):
  learning_rate = 0.0005
  DROP_RATE = 0.3
  CIFAR_10_DIR = 'cifar-10-batches-py'
  BATCH_SIZE = 512
  GPU_RAM_ALLOW_GROWTH = True
  EPOCHS = 35
  SAVE_DIR = 'exp'
  CHECK_POINT = 'nnet'
  CONV_PADDING = 'VALID'
  POOL_PADDING = 'VALID'


class C001(base_config):
  CHECK_POINT = 'nnet_C001'
  ACTIVATION = tf.nn.tanh
  OPTIMIZER = tf.train.AdamOptimizer


class C002(base_config):
  CHECK_POINT = 'nnet_C002'
  ACTIVATION = tf.nn.relu
  OPTIMIZER = tf.train.AdamOptimizer


class C003(base_config):
  CHECK_POINT = 'nnet_C003'
  ACTIVATION = tf.nn.sigmoid
  OPTIMIZER = tf.train.AdamOptimizer


class C004(base_config):
  CHECK_POINT = 'nnet_C004'
  ACTIVATION = tf.nn.relu
  OPTIMIZER = tf.train.GradientDescentOptimizer


PARAM = C004
