import os
import tensorflow as tf
from tensorflow.python.platform import gfile
from model.YOLOFLY import YOLOFLY
from model.DarkNet53 import DarkNet53
from model.LoadWeight import darknet53_load_weights
from model.BaseNet import tiny_darknet_load_weights


class YOLOV3(object):

    def __init__(self, max_box_per_image, warm_up_batches, config, flags):
        self.config = config
        self.flags = flags
        self.max_box_per_image = max_box_per_image
        self.load_weights = None

        self.architecture = config['model']['architecture']
        print('architecture = {}'.format(self.architecture))
        if self.architecture == 'DarkNet53':
            self.loss, self.output_nodes = DarkNet53(max_box_per_image, warm_up_batches, config)
            if flags.init_weights:
                cutoff = 75 if flags.init_weights.endswith('darknet53.conv.74') else 106
                self.load_weights = darknet53_load_weights(prefix='', cutoff=cutoff)
        elif self.architecture == 'YOLOFLY':
            self.loss, self.output_nodes = YOLOFLY(max_box_per_image, warm_up_batches, config)
            self.load_weights = tiny_darknet_load_weights
        else:
            raise Exception('unsupported architecture:{}'.format(self.architecture))

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        self.global_step = tf.train.get_or_create_global_step()
        # self.learning_rate = tf.train.exponential_decay(learning_rate=self.config['train']['learning_rate'],
        #                                                 global_step=self.global_step,
        #                                                 decay_steps=config['train']['epoch_size'] * 10,
        #                                                 decay_rate=0.95,
        #                                                 staircase=True)
        total_steps = config['train']['epoch_size'] * config['train']['train_epoch']
        boundaries = [int(total_steps * r) for r in [0.3, 0.6, 0.8]]
        values = [config['train']['learning_rate'] * r for r in [1., .1, .02, .01]]
        self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries=boundaries, values=values)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # accumulated gradient normalization
        train_vars = tf.trainable_variables()
        accu_vars = [tf.Variable(tf.zeros_like(v.initialized_value()), trainable=False) for v in train_vars]
        self.zero_ops = [accu.assign(tf.zeros_like(v)) for accu, v in zip(accu_vars, train_vars)]
        grad_pairs = self.optimizer.compute_gradients(self.loss, train_vars)
        self.accu_ops = [accu.assign_add(grad / self.config['train']['accumulate'])
                         for (accu, (grad, var)) in zip(accu_vars, grad_pairs)]

        self.train_op = self.optimizer.apply_gradients(
                [(accu, var) for accu, (grad, var) in zip(accu_vars, grad_pairs)],
                global_step=self.global_step
                )

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        # self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def init_weight(self):
        if self.load_weights is None:
            ops = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        elif self.flags.local_mode:
            f = open(self.flags.init_weights)
            ops = self.load_weights(f.read())
        else:
            # distributed training path
            oss_file = os.path.join(self.flags.buckets, self.flags.init_weights)
            f = gfile.GFile(oss_file, mode='r')
            ops = self.load_weights(f.read())
        return ops

    def restore_model(self, sess, saver):
        init_model_dir = self.flags.init_model_dir
        if not self.flags.local_mode:
            init_model_dir = os.path.join(self.flags.buckets, self.flags.init_model_dir)
        print('init_model_dir:{}'.format(init_model_dir))
        ckpt = tf.train.get_checkpoint_state(init_model_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('restore model success.')
