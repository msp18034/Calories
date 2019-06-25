import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers.layers import conv2d
from tensorflow.contrib.layers.python.layers.layers import batch_norm
from tensorflow.contrib.layers.python.layers.layers import max_pool2d


class WeightReader(object):
    def __init__(self, weights_str):
        self.offset = 4
        self.all_weights = np.fromstring(weights_str, dtype=np.float32)
        print(self.all_weights.shape)

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def reset(self):
        self.offset = 4


def tiny_darknet_load_weights(weights_str):

    weight_reader = WeightReader(weights_str)
    weight_reader.reset()
    nb_conv = 16
    i = 1
    ops = []
    graph = tf.get_default_graph()
    while True:
        try:
            t_kernel = graph.get_tensor_by_name('layer_{}/weights:0'.format(i))
        except Exception as e:
            print('get tensor by name failed:layer_{}/weights:0'.format(i))
            break

        if i < nb_conv:
            t_beta = graph.get_tensor_by_name('layer_{}/BatchNorm/beta:0'.format(i))
            t_gamma = graph.get_tensor_by_name('layer_{}/BatchNorm/gamma:0'.format(i))
            t_mean = graph.get_tensor_by_name('layer_{}/BatchNorm/moving_mean:0'.format(i))
            t_var = graph.get_tensor_by_name('layer_{}/BatchNorm/moving_variance:0'.format(i))

            size = np.prod(t_beta.shape.as_list())

            beta = weight_reader.read_bytes(size)
            gamma = weight_reader.read_bytes(size)
            mean = weight_reader.read_bytes(size)
            var = weight_reader.read_bytes(size)

            op_beta = tf.assign(t_beta, beta)
            op_gamma = tf.assign(t_gamma, gamma)
            op_mean = tf.assign(t_mean, mean)
            op_var = tf.assign(t_var, var)

            ops += [op_beta, op_gamma, op_mean, op_var]

        else:
            break

        kernel = weight_reader.read_bytes(np.prod(t_kernel.shape.as_list()))
        kernel = kernel.reshape(list(reversed(t_kernel.shape.as_list())))
        kernel = kernel.transpose([2, 3, 1, 0])
        op_kernel = tf.assign(t_kernel, kernel)

        ops += [op_kernel]

        i += 1
    print('Tiny Weights Loaded.')
    return ops


def BaseNet(input_image, normalizer_params):
    # Layer 1
    x = conv2d(input_image, 16, 3, stride=1, padding='SAME', scope='layer_1',
               activation_fn=tf.nn.leaky_relu,
               normalizer_fn=batch_norm,
               normalizer_params=normalizer_params)
    x = max_pool2d(x, 2, stride=2)
    # Layer 2
    x = conv2d(x, 32, 3, stride=1, padding='SAME', scope='layer_2',
               activation_fn=tf.nn.leaky_relu,
               normalizer_fn=batch_norm,
               normalizer_params=normalizer_params)
    x = max_pool2d(x, 2, stride=2)


    # Layer 3
    x = conv2d(x, 16, 1, stride=1, padding='SAME', scope='layer_3',
               activation_fn=tf.nn.leaky_relu,
               normalizer_fn=batch_norm,
               normalizer_params=normalizer_params)

    # Layer 4
    x = conv2d(x, 128, 3, stride=1, padding='SAME', scope='layer_4',
               activation_fn=tf.nn.leaky_relu,
               normalizer_fn=batch_norm,
               normalizer_params=normalizer_params)

    # Layer 5
    x = conv2d(x, 16, 1, stride=1, padding='SAME', scope='layer_5',
               activation_fn=tf.nn.leaky_relu,
               normalizer_fn=batch_norm,
               normalizer_params=normalizer_params)

    # Layer 6
    x = conv2d(x, 128, 3, stride=1, padding='SAME', scope='layer_6',
               activation_fn=tf.nn.leaky_relu,
               normalizer_fn=batch_norm,
               normalizer_params=normalizer_params)
    x = max_pool2d(x, 2, stride=2)

    # Layer 7
    x = conv2d(x, 32, 1, stride=1, padding='SAME', scope='layer_7',
               activation_fn=tf.nn.leaky_relu,
               normalizer_fn=batch_norm,
               normalizer_params=normalizer_params)

    # Layer 8
    x = conv2d(x, 256, 3, stride=1, padding='SAME', scope='layer_8',
               activation_fn=tf.nn.leaky_relu,
               normalizer_fn=batch_norm,
               normalizer_params=normalizer_params)

    # Layer 9
    x = conv2d(x, 32, 1, stride=1, padding='SAME', scope='layer_9',
               activation_fn=tf.nn.leaky_relu,
               normalizer_fn=batch_norm,
               normalizer_params=normalizer_params)

    # Layer 10
    x = conv2d(x, 256, 3, stride=1, padding='SAME', scope='layer_10',
               activation_fn=tf.nn.leaky_relu,
               normalizer_fn=batch_norm,
               normalizer_params=normalizer_params)
    skip_10 = x
    x = max_pool2d(x, 2, stride=2)

    x = conv2d(x, 64, 1, stride=1, padding='SAME', scope='layer_11',
               activation_fn=tf.nn.leaky_relu,
               normalizer_fn=batch_norm,
               normalizer_params=normalizer_params)

    # Layer 12
    x = conv2d(x, 512, 3, stride=1, padding='SAME', scope='layer_12',
               activation_fn=tf.nn.leaky_relu,
               normalizer_fn=batch_norm,
               normalizer_params=normalizer_params)

    # Layer 13
    x = conv2d(x, 64, 1, stride=1, padding='SAME', scope='layer_13',
               activation_fn=tf.nn.leaky_relu,
               normalizer_fn=batch_norm,
               normalizer_params=normalizer_params)

    # Layer 14
    x = conv2d(x, 512, 3, stride=1, padding='SAME', scope='layer_14',
               activation_fn=tf.nn.leaky_relu,
               normalizer_fn=batch_norm,
               normalizer_params=normalizer_params)

    # Layer 15
    x = conv2d(x, 128, 1, stride=1, padding='SAME', scope='layer_15',
               activation_fn=tf.nn.leaky_relu,
               normalizer_fn=batch_norm,
               normalizer_params=normalizer_params)
    skip_15 = x
    x = max_pool2d(x, 2, stride=2)

    return [x, skip_15, skip_10]
