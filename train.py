import time
import tensorflow as tf
import cifar10_data_manager as cifar10_reader
from cnn_model import CNN_CLASSIFY
import tensorflow.contrib.slim as slim
import sys
import os
import FLAGS


def train_one_epoch(sess, tr_model):
  tr_loss, i = 0, 0
  while True:
    try:
      _, loss = sess.run(
          [tr_model.train_op, tr_model.cross_entropy_loss]
      )
      tr_loss += loss
      i += 1
    except tf.errors.OutOfRangeError:
      break
  tr_loss /= i
  return tr_loss


def train():
  g = tf.Graph()
  with g.as_default():
    with tf.device('/cpu:0'):
      with tf.name_scope('input'):
        x_batch_tr, y_batch_tr, iter_trainset = cifar10_reader.get_batch_from_trainset_use_tfdata()
    with tf.name_scope('model'):
      tr_model = CNN_CLASSIFY(x_batch_tr, y_batch_tr, CNN_CLASSIFY.train)
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

  for epoch in range(1,FLAGS.PARAM.EPOCHS+1):
    stime = time.time()
    sess.run([iter_trainset.initializer])
    tr_loss = train_one_epoch(sess,tr_model)
    etime = time.time()

    # save ckpt
    ckpt_name = "nnet_iter%d_trloss%.4f_duration%ds" % (epoch, tr_loss, etime - stime)
    ckpt_dir = os.path.join(FLAGS.PARAM.SAVE_DIR, FLAGS.PARAM.CHECK_POINT)
    if not os.path.exists(ckpt_dir):
      os.makedirs(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    tr_model.saver.save(sess, ckpt_path)

    msg = ("Train Iteration %03d: \n"
           "    Train.LOSS %.4f, ckpt(%s) saved, EPOCH DURATION: %.2fs\n") % (
            epoch, tr_loss, ckpt_name, etime - stime)
    tf.logging.info(msg)
    sys.stdout.flush()
  sess.close()
  tf.logging.info("Done training")

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
  train()

if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)
  # mkdir exp && OMP_NUM_THREADS=1 python3 train.py 1 2>&1 | tee exp/nnet_CXXX_train_full.log
