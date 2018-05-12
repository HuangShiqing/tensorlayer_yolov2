import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

from variable import FLAGS, META
from layer import RouteLayer


def tiny_yolo_voc(x):
    network = InputLayer(x, name='input')
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 3, 16],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv1_1')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool,
                        name='pool1')
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 16, 32],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv2_1')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool,
                        name='pool2')
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 32, 64],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv3_1')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool,
                        name='pool3')
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 64, 128],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv4_1')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool,
                        name='pool4')
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 128, 256],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv5_1')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool,
                        name='pool5')
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 256, 512],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv6_1')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', pool=tf.nn.max_pool,
                        name='pool6')
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 512, 1024],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv7_1')
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 1024, 1024],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv7_2')
    network = Conv2dLayer(
        network,
        act=tf.identity,
        shape=[1, 1, 1024, 50],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv7_3')
    network = RouteLayer(network, route=-3)

    y = network.outputs
    network.print_layers()
    return y, network
