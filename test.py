import tensorflow as tf
import cifar10_data_manager as cifar10_reader
from cnn_model import CNN_CLASSIFY
import tensorflow.contrib.slim as slim
import sys
import os
import FLAGS
import time


def test():
  stime = time.time()
  g = tf.Graph()
  with g.as_default():
    with tf.device('/cpu:0'):
      with tf.name_scope('input'):
        features, labels = cifar10_reader.read_testset_to_ram()
        x_p, y_p, x_batch_tr, y_batch_tr, iter_testset = cifar10_reader.get_batch_use_tfdata(features,labels)
    with tf.name_scope('model'):
      model = CNN_CLASSIFY(x_batch_tr, y_batch_tr, CNN_CLASSIFY.infer)
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    g.finalize()

    slim.model_analyzer.analyze_vars(tf.trainable_variables(), print_info=True)
    sys.stdout.flush()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = FLAGS.PARAM.GPU_RAM_ALLOW_GROWTH
    config.allow_soft_placement = False
    sess = tf.Session(config=config)
    sess.run(init)

    ckpt = tf.train.get_checkpoint_state(
        os.path.join(FLAGS.PARAM.SAVE_DIR, FLAGS.PARAM.CHECK_POINT))
    if ckpt and ckpt.model_checkpoint_path:
      tf.logging.info("Restore from " + ckpt.model_checkpoint_path)
      model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      tf.logging.fatal("checkpoint not found.")
      sys.exit(-1)
    g.finalize()

  sess.run(iter_testset.initializer,
           feed_dict={x_p: features,
                      y_p: labels})
  accuracy, i = 0, 0
  while True:
    try:
      acc = sess.run(model.accuracy)
      accuracy += acc
      i += 1
    except tf.errors.OutOfRangeError:
      break
  accuracy /= i
  etime = time.time()

  print('Accuracy: %.2f%%, cost time %.2fs.' % (accuracy*100, etime - stime))


def main(argv):
  if not os.path.exists(FLAGS.PARAM.SAVE_DIR):
    os.makedirs(FLAGS.PARAM.SAVE_DIR)
  print('FLAGS.PARAM:')
  supper_dict = FLAGS.base_config.__dict__
  self_dict = FLAGS.PARAM.__dict__
  self_dict_keys = self_dict.keys()
  for key,val in supper_dict.items():
    if key in self_dict_keys:
      print('%s:%s' % (key,self_dict[key]))
    else:
      print('%s:%s' % (key,val))

  # print('\n'.join(['%s:%s' % item for item in PARAM.supper().__dict__.items()]))
  # print('\n'.join(['%s:%s' % item for item in FLAGS.PARAM.__dict__.items()]))
  test()

if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)
  # OMP_NUM_THREADS=1 python3 test.py '' 2>&1 | tee exp/nnet_CXXX_test_full.log
