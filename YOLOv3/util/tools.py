import os
import json
import tensorflow as tf
from tensorflow.python.platform import gfile


# parse tf-record files
def load_sample(flags, config, file_path_pattern, width=500, height=500, channel=3,
                capacity=5000, min_after_dequeue=100, num_threads=12):
    data_path = tf.train.match_filenames_once(file_path_pattern)
    sample_queue = tf.train.string_input_producer(data_path, num_epochs=None,
                                                  shuffle=True, seed=flags.task_index)
    feature = {'train/label': tf.FixedLenFeature([], tf.string),
               'train/image': tf.FixedLenFeature([], tf.string),
               'train/shape': tf.FixedLenFeature([3], tf.int64)
               }
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(sample_queue)
    features = tf.parse_single_example(serialized_example, features=feature)

    single_label = tf.cast(features['train/label'], tf.string)
    single_image = tf.decode_raw(features['train/image'], tf.uint8)
    single_shape = tf.cast(features['train/shape'], tf.int64)
    single_image = tf.reshape(single_image, shape=[height, width, channel])

    images, annotations, shapes = tf.train.shuffle_batch(
        [single_image, single_label, single_shape],
        batch_size=config['train']['batch_size'],
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        num_threads=num_threads,
        seed=flags.task_index,
        shapes=[[height, width, channel], [], [3]])

    return images, annotations, shapes


# load config file and parse train / validation tf-record files based on local / distributed mode
def get_data_path(flags):
    if flags.local_mode:
        config = json.loads(open(flags.model_conf).read())
        train_file_paths = [os.path.join(flags.local_dir, config['train']['tfrecord'])]
        val_file_paths = [os.path.join(flags.local_dir, config['val']['tfrecord'])]
    else:
        # need to uncomment 'buckets' and 'volumes' parameters for distributed conf and train tf-record path.
        conf_file = os.path.join(flags.buckets, flags.model_conf)
        config = json.loads(gfile.GFile(conf_file, mode='r').read())
        volumes = flags.volumes.split(',')
        train_file_paths = [os.path.join(t, config['train']['tfrecord']) for t in volumes]
        val_file_paths = [os.path.join(t, config['val']['tfrecord']) for t in volumes]
    return config, train_file_paths, val_file_paths


# checkpoint & summary output directory
def get_summary_dir(flags):
    if flags.local_mode:
        print('{} local mode {}'.format('*' * 20, '*' * 20))
        checkpoint_dir = flags.checkpointDir
        summary_dir = flags.summary_dir
    else:
        print('{} distributed mode {}'.format('*' * 20, '*' * 20))
        checkpoint_dir = os.path.join(flags.buckets, flags.checkpointDir)
        summary_dir = os.path.join(flags.buckets, flags.summary_dir)
    return checkpoint_dir, summary_dir