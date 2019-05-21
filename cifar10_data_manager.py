import pickle
import os
import numpy as np
from skimage import io as image_io
import matplotlib.pyplot as plt
import tensorflow as tf
import FLAGS


def read_trainset_to_ram():
  data = []
  labels = []
  for i in range(1, 6):
    file = os.path.join(FLAGS.PARAM.CIFAR_10_DIR,'data_batch_%d' % i)
    with open(file, 'rb') as fo:
      batch_dict = pickle.load(fo, encoding='bytes')
      data.extend(batch_dict[b'data'])
      labels.extend((batch_dict[b'labels']))
  # print(len(data),len(labels))
  data = np.array(data,dtype=np.float32)/255.0
  data = np.reshape(data,[-1,3,32,32])
  data = np.transpose(data,[0,2,3,1])
  labels = np.array(labels,dtype=np.int32)
  # print(data.dtype,labels.dtype)

  # img=image_io.imread('imgtest.png')
  # print(np.shape(img))
  # image_io.imshow(img)

  # image_io.imshow(data[40])

  # plt.show()
  # print(np.shape(data),np.shape(labels))
  return data,labels

def read_testset_to_ram():
  data = []
  labels = []
  file = os.path.join(FLAGS.PARAM.CIFAR_10_DIR,'test_batch')
  with open(file, 'rb') as fo:
    batch_dict = pickle.load(fo, encoding='bytes')
    data.extend(batch_dict[b'data'])
    labels.extend((batch_dict[b'labels']))
  # print(len(data),len(labels))
  data = np.array(data,dtype=np.float32)/255.0
  data = np.reshape(data,[-1,3,32,32])
  data = np.transpose(data,[0,2,3,1])
  labels = np.array(labels,dtype=np.int32)

  return data,labels

def get_batch_use_tfdata(features, labels):
  features_placeholder = tf.placeholder(features.dtype, features.shape)
  labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

  dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
  dataset = dataset.batch(FLAGS.PARAM.BATCH_SIZE)
  iterator = dataset.make_initializable_iterator()
  inputs_batch, labels_batch = iterator.get_next()
  return features_placeholder, labels_placeholder, inputs_batch, labels_batch, iterator

if __name__ == '__main__':
  inputs_np, labels_np = read_trainset_to_ram()
  # features, labels = read_testset_to_ram()
  x_p, y_p, inputs, labels, iterator = get_batch_use_tfdata(inputs_np, labels_np)
  sess = tf.Session()
  _, inputs_, labels_ = sess.run([iterator.initializer, inputs, labels],
                                 feed_dict={x_p: inputs_np,
                                            y_p: labels_np})
  print(np.shape(inputs_))
  image_io.imshow(inputs_[40])
  plt.show()
