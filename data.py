import numpy as np
import os
import sys
import glob
import cv2

import xml.etree.ElementTree as ET
from copy import deepcopy
from numpy.random import permutation as perm

from variable import FLAGS, META


def shuffle():
    batch = FLAGS['batch']
    data = parse()
    size = len(data)

    print('Dataset of {} instance(s)'.format(size))
    if batch > size: FLAGS['batch'] = batch = size
    batch_per_epoch = int(size / batch)

    for i in range(FLAGS['epoch']):
        shuffle_idx = perm(np.arange(size))
        for b in range(batch_per_epoch):
            # yield these
            x_batch = list()
            feed_batch = dict()

            for j in range(b * batch, b * batch + batch):
                train_instance = data[shuffle_idx[j]]
                inp, new_feed = get_batch(train_instance)

                if inp is None:
                    continue
                x_batch += [np.expand_dims(inp, 0)]

                for key in new_feed:
                    new = new_feed[key]
                    old_feed = feed_batch.get(key, np.zeros((0,) + new.shape))
                    feed_batch[key] = np.concatenate([old_feed, [new]])

            x_batch = np.concatenate(x_batch, 0)
            yield x_batch, feed_batch

    print('Finish {} epoch(es)'.format(i + 1))


def get_batch(chunk):
    """
    Takes a chunk of parsed annotations
    returns value for placeholders of net's
    input & loss layer correspond to this chunk
    """
    meta = META
    labels = meta['labels']

    H, W, _ = meta['out_size']
    C, B = meta['classes'], meta['num']
    anchors = meta['anchors']

    # preprocess
    jpg = chunk[0]
    w, h, allobj_ = chunk[1]
    allobj = deepcopy(allobj_)
    path = os.path.join(FLAGS['dataset'], jpg)
    img = preprocess(path, allobj)

    # Calculate regression target
    cellx = 1. * w / W
    celly = 1. * h / H
    for obj in allobj:
        centerx = .5 * (obj[1] + obj[3])  # xmin, xmax
        centery = .5 * (obj[2] + obj[4])  # ymin, ymax
        cx = centerx / cellx
        cy = centery / celly
        if cx >= W or cy >= H:
            return None, None
        obj[3] = float(obj[3] - obj[1]) / w
        obj[4] = float(obj[4] - obj[2]) / h
        obj[3] = np.sqrt(obj[3])
        obj[4] = np.sqrt(obj[4])
        obj[1] = cx - np.floor(cx)  # centerx
        obj[2] = cy - np.floor(cy)  # centery
        obj += [int(np.floor(cy) * W + np.floor(cx))]

    # show(im, allobj, S, w, h, cellx, celly) # unit test

    # Calculate placeholders' values
    probs = np.zeros([H * W, B, C])
    confs = np.zeros([H * W, B])
    coord = np.zeros([H * W, B, 4])
    proid = np.zeros([H * W, B, C])
    prear = np.zeros([H * W, 4])
    for obj in allobj:
        probs[obj[5], :, :] = [[0.] * C] * B
        probs[obj[5], :, labels.index(obj[0])] = 1.
        proid[obj[5], :, :] = [[1.] * C] * B
        coord[obj[5], :, :] = [obj[1:5]] * B
        prear[obj[5], 0] = obj[1] - obj[3] ** 2 * .5 * W  # xleft
        prear[obj[5], 1] = obj[2] - obj[4] ** 2 * .5 * H  # yup
        prear[obj[5], 2] = obj[1] + obj[3] ** 2 * .5 * W  # xright
        prear[obj[5], 3] = obj[2] + obj[4] ** 2 * .5 * H  # ybotxl
        confs[obj[5], :] = [1.] * B

    # Finalise the placeholders' values
    upleft = np.expand_dims(prear[:, 0:2], 1)
    botright = np.expand_dims(prear[:, 2:4], 1)
    wh = botright - upleft
    area = wh[:, :, 0] * wh[:, :, 1]
    upleft = np.concatenate([upleft] * B, 1)
    botright = np.concatenate([botright] * B, 1)
    areas = np.concatenate([area] * B, 1)

    # value for placeholder at in
    # put layer
    inp_feed_val = img
    # value for placeholder at loss layer
    loss_feed_val = {
        'probs': probs, 'confs': confs,
        'coord': coord, 'proid': proid,
        'areas': areas, 'upleft': upleft,
        'botright': botright
    }

    return inp_feed_val, loss_feed_val


def parse(exclusive=False):
    meta = META
    ext = '.parsed'
    ann = FLAGS['annotation']
    if not os.path.isdir(ann):
        msg = 'Annotation directory not found {} .'
        exit('Error: {}'.format(msg.format(ann)))
    print('\n{} parsing {}'.format(meta['model'], ann))
    dumps = pascal_voc_clean_xml(ann, meta['labels'], exclusive)
    return dumps


def pascal_voc_clean_xml(ANN, pick, exclusive=False):
    print('Parsing for {} {}'.format(
        pick, 'exclusively' * int(exclusive)))

    dumps = list()
    cur_dir = os.getcwd()
    os.chdir(ANN)
    annotations = os.listdir('.')
    # annotations = glob.glob(str(annotations) + '*.xml')
    size = len(annotations)

    for i, file in enumerate(annotations):
        # progress bar
        sys.stdout.write('\r')
        percentage = 1. * (i + 1) / size
        progress = int(percentage * 20)
        bar_arg = [progress * '=', ' ' * (19 - progress), percentage * 100]
        bar_arg += [file]
        sys.stdout.write('[{}>{}]{:.0f}%  {}'.format(*bar_arg))
        sys.stdout.flush()

        # actual parsing
        in_file = open(file)
        tree = ET.parse(in_file)
        root = tree.getroot()
        jpg = str(root.find('filename').text)
        imsize = root.find('size')
        w = int(imsize.find('width').text)
        h = int(imsize.find('height').text)
        all = list()

        for obj in root.iter('object'):
            current = list()
            name = obj.find('name').text
            if name not in pick:
                continue

            xmlbox = obj.find('bndbox')
            xn = int(float(xmlbox.find('xmin').text))
            xx = int(float(xmlbox.find('xmax').text))
            yn = int(float(xmlbox.find('ymin').text))
            yx = int(float(xmlbox.find('ymax').text))
            current = [name, xn, yn, xx, yx]
            all += [current]

        add = [[jpg, [w, h, all]]]
        dumps += add
        in_file.close()

    # gather all stats
    stat = dict()
    for dump in dumps:
        all = dump[1][2]
        for current in all:
            if current[0] in pick:
                if current[0] in stat:
                    stat[current[0]] += 1
                else:
                    stat[current[0]] = 1

    print('\nStatistics:')
    for i in stat: print('{}: {}'.format(i, stat[i]))

    print('Dataset size: {}'.format(len(dumps)))

    os.chdir(cur_dir)

    return dumps


def preprocess(im, allobj=None):
    """
    Takes an image, return it as a numpy tensor that is readily
    to be fed into tfnet. If there is an accompanied annotation (allobj),
    meaning this preprocessing is serving the train process, then this
    image will be transformed with random noise to augment training data,
    using scale, translation, flipping and recolor. The accompanied
    parsed annotation (allobj) will also be modified accordingly.
    """
    if type(im) is not np.ndarray:
        im = cv2.imread(im)

    if allobj is not None:  # in training mode
        result = imcv2_affine_trans(im)
        im, dims, trans_param = result
        scale, offs, flip = trans_param
        for obj in allobj:
            _fix(obj, dims, scale, offs)
            if not flip: continue
            obj_1_ = obj[1]
            obj[1] = dims[0] - obj[3]
            obj[3] = dims[0] - obj_1_
        im = imcv2_recolor(im)

    im = resize_input(im)
    if allobj is None: return im
    return im  # , np.array(im) # for unit testing


def imcv2_affine_trans(im):
    # Scale and translate
    h, w, c = im.shape
    scale = np.random.uniform() / 10. + 1.
    max_offx = (scale - 1.) * w
    max_offy = (scale - 1.) * h
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)

    im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
    im = im[offy: (offy + h), offx: (offx + w)]
    flip = np.random.binomial(1, .5)
    if flip: im = cv2.flip(im, 1)
    return im, [w, h, c], [scale, [offx, offy], flip]


def _fix(obj, dims, scale, offs):
    for i in range(1, 5):
        dim = dims[(i + 1) % 2]
        off = offs[(i + 1) % 2]
        obj[i] = int(obj[i] * scale - off)
        obj[i] = max(min(obj[i], dim), 0)


def imcv2_recolor(im, a=.1):
    t = [np.random.uniform()]
    t += [np.random.uniform()]
    t += [np.random.uniform()]
    t = np.array(t) * 2. - 1.

    # random amplify each channel
    im = im * (1 + t * a)
    mx = 255. * (1 + a)
    up = np.random.uniform() * 2 - 1
    # 	im = np.power(im/mx, 1. + up * .5)
    im = cv2.pow(im / mx, 1. + up * .5)
    return np.array(im * 255., np.uint8)


def resize_input(im):
    h, w, c = META['inp_size']
    imsz = cv2.resize(im, (w, h))
    imsz = imsz / 255.
    imsz = imsz[:, :, ::-1]
    return imsz
