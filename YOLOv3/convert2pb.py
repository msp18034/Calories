#! /usr/bin/env python

import json
import tensorflow as tf
from network import YOLOV3
from tensorflow.contrib.quantize.python.fold_batch_norms import FoldBatchNorms


tf.app.flags.DEFINE_string('conf', None, 'config file')
tf.app.flags.DEFINE_string('model_dir', None, 'tf model directory')
tf.app.flags.DEFINE_string('output_pb', None, 'model pb')
tf.app.flags.DEFINE_boolean('local_mode', True, 'define if is local mode')
tf.app.flags.DEFINE_string('init_weights', None, 'init weights')
tf.app.flags.DEFINE_boolean('restore', True, 'restore or initialize')
flags = tf.app.flags.FLAGS


def convert(args):
    with open(flags.conf) as config_buffer:
        config = json.load(config_buffer)

    model = YOLOV3(max_box_per_image=config['train']['max_box_per_image'],
                   warm_up_batches=config['train']['warm_up_epoch'] * config['train']['epoch_size'],
                   config=config, flags=flags)

    sess = tf.Session()
    if flags.restore:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(flags.model_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        init_ops = tf.global_variables_initializer()
        assign_ops = model.init_weight()
        sess.run(init_ops)
        sess.run(assign_ops)

    sess.graph.as_default()

    if config['model']['architecture'] in ('YOLOFLY',):
        output_node_names = "layer_29/BiasAdd,layer_25/BiasAdd,layer_21/BiasAdd"
    elif config['model']['architecture'] in ('DarkNet53',):
        output_node_names = "conv_81/BiasAdd,conv_93/BiasAdd,conv_105/BiasAdd"
    else:
        raise Exception('unknow model architecture:{}'.format(config['model']['architecture']))

    subgraph = tf.graph_util.extract_sub_graph(tf.get_default_graph().as_graph_def(), output_node_names.split(','))
    tf.reset_default_graph()
    tf.import_graph_def(subgraph)

    if tf.VERSION.strip().split('-')[0] > '1.5.0':
        FoldBatchNorms(tf.get_default_graph(), False)  # not suitable for tensorflow 1.4

    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        subgraph,
        output_node_names.split(",")
    )

    with tf.gfile.GFile(flags.output_pb, 'wb') as f:
        f.write(output_graph_def.SerializeToString())


if __name__ == '__main__':
    tf.app.run(convert)
