#! /usr/bin/env python
import os
import tensorflow as tf
from tensorflow.python.framework import ops
from network import YOLOV3
from util.augment import Augment
from util.evaluator import Evaluator
from util.summary_save import SummaryHook
from util.best_saver import BestSaverHook
from util.early_stop import EarlyStopHook
from util.checkpoint_saver_listener import BestCheckpointSaverListener
from util.tools import load_sample, get_data_path, get_summary_dir

'''
you need to rewrite these two or define more parameters based your specific distributed tf platform
to specify training data, config file, checkpoint model storage path, etc.
'''
# tf.app.flags.DEFINE_string('volumes', '', 'tf-record data path for distributed training / validation')
# tf.app.flags.DEFINE_string('buckets', None, 'oss buckets path for model and config storage during distributed train')


tf.app.flags.DEFINE_string('ps_hosts', '', 'ps hosts')
tf.app.flags.DEFINE_string('worker_hosts', '', 'worker hosts')
tf.app.flags.DEFINE_string('job_name', None, 'worker or ps')
tf.app.flags.DEFINE_string('checkpointDir', None, 'model directory')
tf.app.flags.DEFINE_string('summary_dir', 'log', 'summary directory')
tf.app.flags.DEFINE_string('model_conf', None, 'model conf file')
tf.app.flags.DEFINE_boolean('restore', False, 'restore from pre-train model or not')
tf.app.flags.DEFINE_string('init_weights', 'conf/init/tiny.weights', 'init weights')
tf.app.flags.DEFINE_string('init_model_dir', None, 'pre-train model directory')
tf.app.flags.DEFINE_integer('task_index', None, 'worker task index')
tf.app.flags.DEFINE_boolean('local_mode', False, 'local or distributed mode')
tf.app.flags.DEFINE_string('local_dir', 'data/voc', 'local default train file path')
tf.app.flags.DEFINE_boolean('resume', False, 'resume to train or not')

flags = tf.app.flags.FLAGS


def create_model(config):
    if flags.restore:
        config['train']['warm_up_epoch'] = 0
    model = YOLOV3(max_box_per_image=config['train']['max_box_per_image'],
                   warm_up_batches=config['train']['warm_up_epoch'] * config['train']['epoch_size'],
                   config=config, flags=flags)
    return model


def create_hook(config, info_dict):
    checkpoint_dir, summary_dir = get_summary_dir(flags)

    var_list = [v for v in tf.global_variables() if 'global_step' not in v.name]
    if flags.resume:
        var_list = None
    restore_saver = tf.train.Saver(var_list=var_list)

    # chief summary hook
    lr = tf.summary.scalar('lr', info_dict['lr'])
    train_loss = tf.summary.scalar('train_loss', info_dict['train_loss'])

    summary_writer = tf.summary.FileWriter(summary_dir)
    summary_op = [train_loss, lr]
    for v in tf.get_collection(tf.GraphKeys.SUMMARIES):
        name = os.path.basename(v.name)
        if name.startswith('loss_'):
            summary_op.append(v)
    summary_op = tf.summary.merge(summary_op)
    summary_hook = SummaryHook(save_steps=config['train']['summary_step'],
                               summary_writer=summary_writer,
                               summary_op=summary_op)
    if flags.task_index == 0:
        summary_writer.add_graph(graph=ops.get_default_graph())

    # stop hook
    last_step = config['train']['epoch_size'] * (config['train']['train_epoch'] + config['train']['warm_up_epoch'])
    stop_hook = tf.train.StopAtStepHook(last_step=last_step)
    early_stop_hook = EarlyStopHook()

    # chief checkpoint save hook
    saver = tf.train.Saver(max_to_keep=1)
    best_saver = tf.train.Saver(max_to_keep=1)
    model_path = os.path.join(checkpoint_dir, 'best/best_' + config['model']['name'])
    print('checkpoint_dir:{}\nsummary_dir:{}\nbest_model_dir:{}'.format(checkpoint_dir, summary_dir, model_path))

    # early stop default turn off
    saver_listener = BestCheckpointSaverListener(saver=best_saver, config=config, validator=info_dict['validator'],
                                                 model_path=model_path, summary_writer=summary_writer,
                                                 early_stop=False, patience=10)
    saver_hook = BestSaverHook(checkpoint_dir=checkpoint_dir, save_steps=config['train']['epoch_size'],
                               saver=saver, checkpoint_basename=config['model']['name'],
                               listeners=[saver_listener])

    hooks = [stop_hook, early_stop_hook]
    chief_only_hooks = [saver_hook, summary_hook]
    hook_dict = {'hooks': hooks,
                 'chief_only_hooks': chief_only_hooks,
                 'dependent_ops': [summary_op],
                 'restore_saver': restore_saver}
    return hook_dict


def train(server, cluster):
    is_chief = (flags.task_index == 0)
    config, train_file_path, val_file_path = get_data_path(flags)
    tf_height, tf_width, tf_channel = config['model']['tf_input_size']
    accumulate = config['train']['accumulate'] if 'accumulate' in config['train'] else 8

    worker_device = '/job:worker/task:{}'.format(flags.task_index)
    with tf.device(worker_device):
        train_images, train_annotations, train_shapes = load_sample(flags, config, train_file_path, width=tf_width,
                                                                    height=tf_height, channel=tf_channel)
        val_images, val_annotations, val_shapes = load_sample(flags, config, val_file_path,
                                                              width=tf_width, height=tf_height, channel=tf_channel,
                                                              capacity=500, min_after_dequeue=10, num_threads=2)

    augment = Augment(config)
    with tf.device('/job:worker/task:{}/cpu:0'.format(flags.task_index)):
        train_input = tf.py_func(augment, [train_images, train_annotations, train_shapes, True],
                                 [tf.float32] * 5, stateful=True)
        val_input = tf.py_func(augment, [val_images, val_annotations, val_shapes, False],
                               [tf.float32] * 5 + [tf.uint8, tf.string, tf.int64], stateful=False)

    with tf.device(tf.train.replica_device_setter(worker_device=worker_device, cluster=cluster)):
        model = create_model(config)
        if is_chief and not flags.restore:
            init_ops = model.init_weight()

    validator = Evaluator(config, augment,
                          {'loss': model.loss, 'data': val_input, 'output_nodes': model.output_nodes,
                           'nms_func': 'fast_nms', 'cls_func': 'softmax'})
    hook_dict = create_hook(config, {'lr': model.learning_rate, 'train_loss': model.loss, 'validator': validator})
    total_ops = hook_dict['dependent_ops'] + [model.accu_ops, model.update_ops, model.loss, model.global_step]

    with tf.train.MonitoredTrainingSession(master=server.target, hooks=hook_dict['hooks'],
                                           chief_only_hooks=hook_dict['chief_only_hooks'], is_chief=is_chief) as sess:
        if is_chief and not flags.restore:
            sess.run(init_ops)
        elif is_chief and flags.restore:
            model.restore_model(sess, hook_dict['restore_saver'])
        while not sess.should_stop():
            for _ in range(accumulate):
                if not sess.should_stop():
                    t_input = sess._tf_sess().run(train_input)
                if not sess.should_stop():
                    sess.run(total_ops,
                             feed_dict={'input_image:0': t_input[0],
                                        'true_boxes:0': t_input[1],
                                        'true_yolo_1:0': t_input[2],
                                        'true_yolo_2:0': t_input[3],
                                        'true_yolo_3:0': t_input[4],
                                        'phase:0': True,
                                        }
                             )

            if not sess.should_stop():
                sess.run([model.train_op, model.global_step])

            if not sess.should_stop():
                sess._tf_sess().run([model.zero_ops, model.global_step])


def main(args):
    print('{} flag parameters {}'.format('*' * 20, '*' * 20))
    for key, value in flags.__flags.items():
        print(key, value)

    cluster_spec = {}
    ps_spec = flags.ps_hosts.split(',')
    cluster_spec['ps'] = ps_spec
    worker_spec = flags.worker_hosts.split(',')
    cluster_spec['worker'] = worker_spec
    cluster = tf.train.ClusterSpec(cluster_spec)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    server = tf.train.Server(cluster, job_name=flags.job_name, config=config, task_index=flags.task_index)

    if flags.job_name == 'ps':
        server.join()
    elif flags.job_name == 'worker':
        train(server=server, cluster=cluster)


if __name__ == '__main__':
    tf.app.run(main)
