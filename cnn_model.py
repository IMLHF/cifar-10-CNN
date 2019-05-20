# author: lihongfeng 2019-5-20
# github:
import tensorflow as tf
import FLAGS


class CNN_CLASSIFY(object):
    def __init__(self, input_batch, num_label_batch):
        '''
        input_batch: [batch_size,32,32,3]
        label_batch: [batch_size]
        '''
        self._input_batch = input_batch
        self._num_label_batch = num_label_batch

        # onehot_label
        batch_size = tf.size(self._num_label_batch)
        self._onehot_label_batch = tf.expand_dims(self._num_label_batch, 1)
        indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
        concated = tf.concat([indices, self._onehot_label_batch],1)
        self._onehot_label_batch = tf.sparse_to_dense(concated, tf.stack([batch_size, 10]), 1.0, 0.0) # [batch_size,10]


        weights = {
            'w_conv1': tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 32], stddev=0.1)),
            'w_conv2': tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=0.1)),
            'w_fc1': tf.Variable(tf.truncated_normal(shape=[5 * 5 * 64, 1024], stddev=0.1)),
            'w_fc2': tf.Variable(tf.truncated_normal(shape=[1024, 512], stddev=0.1)),
            'w_fc3': tf.Variable(tf.truncated_normal(shape=[512, 10], stddev=0.1)),
        }
        biases={
            'b_conv1': tf.Variable(tf.constant(0.1, shape=[32])),
            'b_conv2': tf.Variable(tf.constant(0.1, shape=[64])),
            'b_fc1': tf.Variable(tf.constant(0.1, shape=[1024])),
            'b_fc2': tf.Variable(tf.constant(0.1, shape=[512])),
            'b_fc3': tf.Variable(tf.constant(0.1, shape=[10]))
        }

        out_conv1 = FLAGS.PARAM.ACTIVATION(
            tf.nn.conv2d(self._input_batch,
                         weights['w_conv1'],
                         [1, 1, 1, 1],
                         padding=FLAGS.PARAM.CONV_PADDING
                         ) + biases['b_conv1'])
        out_mp1 = tf.nn.max_pool(out_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding=FLAGS.PARAM.POOL_PADDING)
        out_conv2 = FLAGS.PARAM.ACTIVATION(
            tf.nn.conv2d(out_mp1,
                         weights['w_conv2'],
                         [1, 1, 1, 1],
                         padding=FLAGS.PARAM.CONV_PADDING
                         ) + biases['b_conv2'])
        out_mp2 = tf.nn.max_pool(out_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding=FLAGS.PARAM.POOL_PADDING)
        out_conv_flatten = tf.reshape(out_mp2,[-1,5*5*64])
        out_fc1 = FLAGS.PARAM.ACTIVATION(
            tf.matmul(out_conv_flatten,weights['w_fc1'])+biases['b_fc1']
        )
        out_drop_fc1 = tf.nn.dropout(out_fc1,keep_prob=1.0-FLAGS.PARAM.DROP_RATE)
        out_fc2 = FLAGS.PARAM.ACTIVATION(
            tf.matmul(out_drop_fc1,weights['w_fc2'] + biases['b_fc2'])
        )
        out_fc3 = FLAGS.PARAM.ACTIVATION(
            tf.matmul(out_fc2,weights['w_fc3'] + biases['b_fc3'])
        )
        self._logits = out_fc3
        self._out_softmax=tf.nn.softmax(self._logits)
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self._logits, labels=self._onehot_label_batch))
        self._train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.PARAM.learning_rate).minimize(cross_entropy_loss)

    @property
    def train_op(self):
        return self._train_op

    @property
    def out_softmax(self):
        return self._out_softmax
