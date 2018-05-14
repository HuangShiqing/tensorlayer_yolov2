import tensorflow as tf
import os

from net import tiny_yolo_voc,yolo
from train import train
from variable import FLAGS, META

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'


def main():
    inp_size = [None] + META['inp_size']
    input_pb = tf.placeholder(tf.float32, inp_size)

    # out, net = tiny_yolo_voc(input_pb)
    out, net = yolo(input_pb)
    exit()

    train(out, input_pb)


if __name__ == '__main__':
    main()
