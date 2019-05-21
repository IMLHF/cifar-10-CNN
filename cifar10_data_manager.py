import pickle
import os
import numpy as np
from skimage import io as image_io
import matplotlib.pyplot as plt
import tensorflow as tf
import FLAGS


def get_batch_from_trainset_use_tfdata():
  data = []
  labels = []
  for i in range(1, 6):
    file = os.path.join(FLAGS.PARAM.CIFAR_10_DIR,'data_batch_%d' % i)
    with open(file, 'rb') as fo:
      batch_dict = pickle.load(fo, encoding='bytes')
      data.extend(batch_dict[b'data'])
      labels.extend((batch_dict[b'labels']))
  # print(len(data),len(labels))
  data = np.array(data)
  data = np.reshape(data,[-1,3,32,32])
  data = np.transpose(data,[0,2,3,1])
  labels = np.array(labels)
  # print(data.dtype,labels.dtype)

  # img=image_io.imread('imgtest.png')
  # print(np.shape(img))
  # image_io.imshow(img)

  # image_io.imshow(data[40])

  # plt.show()
  # print(np.shape(data),np.shape(labels))
  dataset = tf.data.Dataset.from_tensor_slices((data,labels))
  dataset = dataset.batch(FLAGS.PARAM.BATCH_SIZE)
  iterator = dataset.make_initializable_iterator()
  inputs, labels = iterator.get_next()
  return inputs, labels, iterator

def get_batch_from_testset_use_tfdata():
  data = []
  labels = []
  file = os.path.join(FLAGS.PARAM.CIFAR_10_DIR,'test_batch')
  with open(file, 'rb') as fo:
    batch_dict = pickle.load(fo, encoding='bytes')
    data.extend(batch_dict[b'data'])
    labels.extend((batch_dict[b'labels']))
  # print(len(data),len(labels))
  data = np.array(data)
  data = np.reshape(data,[-1,3,32,32])
  data = np.transpose(data,[0,2,3,1])
  labels = np.array(labels)

  dataset = tf.data.Dataset.from_tensor_slices((data,labels))
  dataset = dataset.batch(FLAGS.PARAM.BATCH_SIZE)
  iterator = dataset.make_initializable_iterator()
  inputs, labels = iterator.get_next()
  return inputs, labels, iterator

if __name__ == '__main__':
  # inputs,labels,iterator = get_batch_from_trainset_use_tfdata()
  inputs, labels, iterator = get_batch_from_testset_use_tfdata()
  sess = tf.Session()
  _, inputs_, labels_ = sess.run([iterator.initializer,inputs,labels])
  print(np.shape(inputs_))
  image_io.imshow(inputs_[40])
  plt.show()
