import tensorflow as tf
import numpy as np
import os
import pickle

from variable import FLAGS, META
import data


def expit_tensor(x):
    return 1. / (1. + tf.exp(-x))


def build_train_op(loss):
    # self.framework.loss(self.out)
    # self.say('Building {} train op'.format(self.meta['model']))
    # optimizer = self._TRAINER[self.FLAGS.trainer](self.FLAGS.lr)
    # optimizer = tf.train.AdamOptimizer(FLAGS.lr)
    # gradients = optimizer.compute_gradients(self.framework.loss)
    # train_op = optimizer.apply_gradients(gradients)
    # self.train_op = optimizer.minimize(self.out)

    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS['lr'])
    train_op = optimizer.minimize(loss)

    return train_op


def build_summary_op():
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.summary + 'train')

    return summary_op, writer


def loss_op(net_out):
    """
    Takes net.out and placeholders value
    returned in batch() func above,
    to build train_op and loss
    """
    # meta
    m = META
    sprob = float(m['class_scale'])
    sconf = float(m['object_scale'])
    snoob = float(m['noobject_scale'])
    scoor = float(m['coord_scale'])
    H, W, _ = m['out_size']
    B, C = m['num'], m['classes']
    HW = H * W  # number of grid cells
    anchors = m['anchors']

    print('{} loss hyper-parameters:'.format(m['model']))
    print('\tH       = {}'.format(H))
    print('\tW       = {}'.format(W))
    print('\tbox     = {}'.format(m['num']))
    print('\tclasses = {}'.format(m['classes']))
    print('\tscales  = {}'.format([sprob, sconf, snoob, scoor]))

    size1 = [None, HW, B, C]
    size2 = [None, HW, B]

    # return the below placeholders
    _probs = tf.placeholder(tf.float32, size1)
    _confs = tf.placeholder(tf.float32, size2)
    _coord = tf.placeholder(tf.float32, size2 + [4])
    # weights term for L2 loss
    _proid = tf.placeholder(tf.float32, size1)
    # material calculating IOU
    _areas = tf.placeholder(tf.float32, size2)
    _upleft = tf.placeholder(tf.float32, size2 + [2])
    _botright = tf.placeholder(tf.float32, size2 + [2])

    placeholders = {
        'probs': _probs, 'confs': _confs, 'coord': _coord, 'proid': _proid,
        'areas': _areas, 'upleft': _upleft, 'botright': _botright
    }

    # Extract the coordinate prediction from net.out
    net_out_reshape = tf.reshape(net_out, [-1, H, W, B, (4 + 1 + C)])
    coords = net_out_reshape[:, :, :, :, :4]
    coords = tf.reshape(coords, [-1, H * W, B, 4])
    adjusted_coords_xy = expit_tensor(coords[:, :, :, 0:2])
    adjusted_coords_wh = tf.sqrt(
        tf.exp(coords[:, :, :, 2:4]) * np.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2]))
    coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

    adjusted_c = expit_tensor(net_out_reshape[:, :, :, :, 4])
    adjusted_c = tf.reshape(adjusted_c, [-1, H * W, B, 1])

    adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 5:])
    adjusted_prob = tf.reshape(adjusted_prob, [-1, H * W, B, C])

    adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3)

    wh = tf.pow(coords[:, :, :, 2:4], 2) * np.reshape([W, H], [1, 1, 1, 2])
    area_pred = wh[:, :, :, 0] * wh[:, :, :, 1]
    centers = coords[:, :, :, 0:2]
    floor = centers - (wh * .5)
    ceil = centers + (wh * .5)

    # calculate the intersection areas
    intersect_upleft = tf.maximum(floor, _upleft)
    intersect_botright = tf.minimum(ceil, _botright)
    intersect_wh = intersect_botright - intersect_upleft
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect = tf.multiply(intersect_wh[:, :, :, 0], intersect_wh[:, :, :, 1])

    # calculate the best IOU, set 0.0 confidence for worse boxes
    iou = tf.truediv(intersect, _areas + area_pred - intersect)
    best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
    best_box = tf.to_float(best_box)
    confs = tf.multiply(best_box, _confs)

    # take care of the weight terms
    conid = snoob * (1. - confs) + sconf * confs
    weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)
    cooid = scoor * weight_coo
    weight_pro = tf.concat(C * [tf.expand_dims(confs, -1)], 3)
    proid = sprob * weight_pro

    # fetch += [_probs, confs, conid, cooid, proid]
    true = tf.concat([_coord, tf.expand_dims(confs, 3), _probs], 3)
    wght = tf.concat([cooid, tf.expand_dims(conid, 3), proid], 3)

    print('Building {} loss'.format(m['model']))
    loss = tf.pow(adjusted_net_out - true, 2)
    loss = tf.multiply(loss, wght)
    loss = tf.reshape(loss, [-1, H * W * B * (4 + 1 + C)])
    loss = tf.reduce_sum(loss, 1)
    loss = .5 * tf.reduce_mean(loss)
    tf.summary.scalar('{} loss'.format(m['model']), loss)

    return loss, placeholders


train_stats = (
    'Training statistics: \n'
    '\tLearning rate : {}\n'
    '\tBatch size    : {}\n'
    '\tEpoch number  : {}\n'
    '\tBackup every  : {}'
)


def train(net_out, input_pb):
    batches = data.shuffle()
    loss, loss_ph = loss_op(net_out)
    train_op = build_train_op(loss)
    if FLAGS['summary']:
        summary_op, writer = build_summary_op()

    # loss_mva = None;
    profile = list()

    config = tf.ConfigProto()
    # config.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = utility)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        for i, (x_batch, datum) in enumerate(batches):

            if not i: print(train_stats.format(
                FLAGS['lr'], FLAGS['batch'],
                FLAGS['epoch'], FLAGS['save']
            ))

            feed_dict = {
                loss_ph[key]: datum[key]
                for key in loss_ph}
            feed_dict[input_pb] = x_batch
            # feed_dict.update(self.feed)

            fetches = [train_op, loss]

            if FLAGS['summary']:
                fetches.append(summary_op)

            fetched = sess.run(fetches, feed_dict)
            final_loss = fetched[1]

            # if loss_mva is None: loss_mva = loss
            # loss_mva = .9 * loss_mva + .1 * loss
            step_now = FLAGS['load'] + i + 1

            if FLAGS['summary']:
                writer.add_summary(fetched[2], step_now)

            form = 'step {} - loss {}'
            print(form.format(step_now, final_loss))

            profile += [final_loss]

            ckpt = (i + 1) % FLAGS['save']
            args = [sess, step_now, profile]
            if not ckpt:
                save_ckpt(*args)

        if ckpt:
            save_ckpt(*args)


def save_ckpt(sess, step, loss_profile):
    file = '{}-{}{}'
    model = META['name']

    profile = file.format(model, step, '.profile')
    profile = os.path.join(FLAGS.backup, profile)
    with open(profile, 'wb') as profile_ckpt:
        pickle.dump(loss_profile, profile_ckpt)

    ckpt = file.format(model, step, '')
    ckpt = os.path.join(FLAGS.backup, ckpt)

    print('Checkpoint at step {}'.format(step))

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
    saver.save(sess, ckpt)
