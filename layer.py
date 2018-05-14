import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

from tensorlayer.deprecation import deprecated_alias
import tensorlayer._logging as logging


class RouteLayer(Layer):
    @deprecated_alias(layer='prev_layer', end_support_version=1.9)
    def __init__(self, prev_layer=None, routes=None, name='routes'):
        super(RouteLayer, self).__init__(prev_layer=prev_layer, name=name)

        for route in routes:
            if abs(route) >= len(prev_layer.all_layers):
                raise Exception("beyond the num of layers")

        logging.info("RouteLayer  %s: routes:%s" % (name, str(routes)))

        self.inputs = prev_layer.outputs

        out = list()
        for i, route in enumerate(routes):
            out.append(prev_layer.all_layers[route])

        with tf.variable_scope(name):
            self.outputs = tf.concat([o for o in out], 3)

        self.all_layers.append(self.outputs)


class ReorgLayer(Layer):
    @deprecated_alias(layer='prev_layer', end_support_version=1.9)
    def __init__(self, prev_layer=None, reorg=None, name='reorg'):
        super(ReorgLayer, self).__init__(prev_layer=prev_layer, name=name)

        out = list()
        logging.info("ReorgLayer  %s: reorg:%s" % (name, str(reorg)))

        self.inputs = prev_layer.outputs

        with tf.variable_scope(name):
            self.outputs = tf.space_to_depth(self.inputs, reorg)

        self.all_layers.append(self.outputs)

# class RouteLayer(Layer):
#     @deprecated_alias(layer='prev_layer', end_support_version=1.9)
#     def __init__(self, prev_layer=None, routes=None, name='routes'):
#         super(ReorgLayer, self).__init__(prev_layer=prev_layer, name=name)
#
#         routes.insert(0, -1)
#         out = list()
#
#         for route in routes:
#             if abs(route) >= len(prev_layer.all_layers):
#                 raise Exception("beyond the num of layers")
#
#         logging.info("RouteLayer  %s: routes:%s" % (name, str(routes)))
#
#         self.inputs = prev_layer.outputs
#
#         for i, route in enumerate(routes):
#             out.append(prev_layer.all_layers[route])
#
#         for i, route in enumerate(out):
#             if i > 0:
#                 out[i] = tf.space_to_depth(out[i], out[i].get_shape()[1].value / out[0].get_shape()[1].value)
#
#         with tf.variable_scope(name):
#             self.outputs = tf.concat([o for o in out], 3)
#
#         self.all_layers.append(self.outputs)
