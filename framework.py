import data
import train




class YOLOv2(object):
    def __init__(self):
        shuffle = data.shuffle
        loss_op = train.loss

        train = train.train
