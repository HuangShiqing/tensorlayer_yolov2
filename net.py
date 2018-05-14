import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

from variable import FLAGS, META
from layer import RouteLayer, ReorgLayer


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
    network = RouteLayer(network, routes=[-7])
    y = network.outputs
    network.print_layers()
    return y, network


def yolo(x):
    network = InputLayer(x, name='input')
    # 0
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 3, 32],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv1_1')
    #     network = BatchNormLayer(network,is_train=True,name='BN_1')
    # 1
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool,
                        name='pool1')
    # 2
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 32, 64],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv2_1')
    #     network = BatchNormLayer(network,is_train=True,name='BN_1')
    # 3
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool,
                        name='pool2')
    # 4
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 64, 128],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv3_1')
    # 5
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[1, 1, 128, 64],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv3_2')
    # 6
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 64, 128],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv3_3')
    #     network = BatchNormLayer(network,is_train=True,name='BN_1')
    # 7
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool,
                        name='pool3')
    # 8
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 128, 256],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv4_1')
    # 9
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[1, 1, 256, 128],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv4_2')
    # 10
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 128, 256],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv4_3')
    # 11
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool,
                        name='pool4')
    # 12
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 256, 512],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv5_1')
    # 13
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[1, 1, 512, 256],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv5_2')
    # 14
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 256, 512],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv5_3')
    # 15
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[1, 1, 512, 256],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv5_4')
    # 16
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 256, 512],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv5_5')
    # 17
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool,
                        name='pool5')
    # 18
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 512, 1024],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv6_1')
    # 19
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[1, 1, 1024, 512],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv6_2')
    # 20
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 512, 1024],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv6_3')
    # 21
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[1, 1, 1024, 512],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv6_4')
    # 22
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 512, 1024],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv6_5')
    # network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool,
    #                     name='pool6')
    # 23
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 1024, 1024],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv7_1')
    # 24
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 1024, 1024],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv7_2')
    # 25
    network = RouteLayer(network, routes=[-9])
    # 26
    # network = Conv2dLayer(
    #     network,
    #     act=tf.nn.relu,
    #     shape=[1, 1, 512, 64],  # 64 features for each 3x3 patch
    #     strides=[1, 1, 1, 1],
    #     padding='SAME',
    #     name='conv8_1')
    network = ReorgLayer(network, reorg=2)
    network = RouteLayer(network, routes=[-1, -3])
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 3072, 1024],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv9_1')
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[1, 1, 1024, 50],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv9_2')

    y = network.outputs
    network.print_layers()
    return y, network
