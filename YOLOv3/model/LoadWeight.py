import struct
import numpy as np
import tensorflow as tf


class WeightReader(object):
    def __init__(self, weights_str):
        self.offset = 3
        major = struct.unpack('i', weights_str[0:4])[0]
        minor = struct.unpack('i', weights_str[4:8])[0]
        revision = struct.unpack('i', weights_str[8:12])[0]
        if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
            self.offset += 2
        else:
            self.offset += 1
        print('skip first {} int32 header'.format(self.offset))
        self.all_weights = np.fromstring(weights_str[self.offset * 4:], dtype=np.float32)
        self.offset = 0

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def reset(self):
        self.offset = 0


def darknet53_load_weights(prefix='', cutoff=106):

    def load_weights(weights_str, pre=prefix, cut=cutoff):
        weight_reader = WeightReader(weights_str)
        ops = []
        graph = tf.get_default_graph()
        for i in range(cut):
            try:
                t_kernel = graph.get_tensor_by_name('{}conv_{}/kernel:0'.format(pre, i))
            except Exception as e:
                # print('no layer layer_conv_{}'.format(i))
                continue

            if i not in [81, 93, 105]:
                t_beta = graph.get_tensor_by_name('{}norm_{}/beta:0'.format(pre, i))
                t_gamma = graph.get_tensor_by_name('{}norm_{}/gamma:0'.format(pre, i))
                t_mean = graph.get_tensor_by_name('{}norm_{}/moving_mean:0'.format(pre, i))
                t_var = graph.get_tensor_by_name('{}norm_{}/moving_variance:0'.format(pre, i))

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
                t_bias = graph.get_tensor_by_name("{}conv_{}/bias:0".format(pre, i))
                size = np.prod(t_bias.shape.as_list())
                bias = weight_reader.read_bytes(size)

                op_bias = tf.assign(t_bias, bias)
                ops += [op_bias]

            kernel = weight_reader.read_bytes(np.prod(t_kernel.shape.as_list()))
            kernel = kernel.reshape(list(reversed(t_kernel.shape.as_list())))
            kernel = kernel.transpose([2, 3, 1, 0])
            op_kernel = tf.assign(t_kernel, kernel)

            ops += [op_kernel]

        print('DarkNet53 Weights Loaded.')
        return ops

    return load_weights




