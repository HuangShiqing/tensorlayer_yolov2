

FLAGS = {'save': 2000, 'labels': 'labels.txt', 'batch': 16, 'gpu': 0.95, 'momentum': 0.0,
         'dataset': '/home/hsq/DeepLearning/data/car/bdd100k/images/100k/train/', 'backup': './ckpt/',
         'binary': './bin/', 'summary': '',
         'annotation': '/home/hsq/DeepLearning/data/car/bdd100k/labels/100k/train_xml/', 'pbLoad': '',
         'metaLoad': '', 'config': './cfg/', 'verbalise': True, 'model': 'cfg/tiny-yolo-voc-1c.cfg',
         'json': False, 'queue': 1, 'lr': 1e-05, 'saveVideo': False, 'train': False, 'trainer': 'adam',
         'load': 0, 'threshold': -0.1, 'gpuName': '/gpu:0', 'epoch': 100, 'savepb': False,
         'imgdir': './sample_img/', 'demo': '', 'keep': 5}

META = {'class_scale': 1, 'num': 5, 'model': 'cfg/tiny-yolo-voc-1c.cfg',
        'colors': [(254.0, 254.0, 254), (222.25, 190.5, 127), (190.5, 127.0, 254),
                   (158.75, 63.5, 127), (127.0, 254.0, 254)], 'net': {'hue': 0.1, 'width': 416,
                                                                      'saturation': 1.5, 'batch': 64,
                                                                      'subdivisions': 8, 'policy': 'steps',
                                                                      'max_batches': 40100, 'momentum': 0.9,
                                                                      'height': 416, 'exposure': 1.5,
                                                                      'learning_rate': 0.001, 'type': '[net]',
                                                                      'steps': '-1,100,20000,30000',
                                                                      'decay': 0.0005, 'scales': '.1,10,.1,.1',
                                                                      'channels': 3, 'angle': 0},
        'name': 'tiny-yolo-voc-1c', 'object_scale': 5, 'absolute': 1, 'rescore': 1, 'inp_size': [416, 416, 3],
        'coords': 4, 'random': 1, 'out_size': [13, 13, 50], 'bias_match': 1, 'jitter': 0.2, 'type': '[region]',
        'thresh': 0.5, 'classes': 5, 'anchors': [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52],
        'softmax': 1, 'labels': ['bus', 'truck', 'motor', 'car', 'train'], 'noobject_scale': 1, 'coord_scale': 1}

