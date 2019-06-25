import os
import cv2
import json
import numpy as np
import tensorflow as tf
from multiprocessing import Process
from multiprocessing import Queue

from gen_anchors import parse_voc_annotation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.flags.DEFINE_string('anno_folder', None, 'annotation directory')
tf.flags.DEFINE_string('image_folder', None, 'image directory')
tf.flags.DEFINE_string('name_file', '', 'image name file')
tf.flags.DEFINE_string('output_path', None, 'tf-record output file path')
tf.flags.DEFINE_integer('norm_width', None, 'tf-record image width')
tf.flags.DEFINE_integer('norm_height', None, 'tf-record image height')
tf.flags.DEFINE_integer('num_process', 4, 'number of process')
tf.flags.DEFINE_integer('max_file_size', 1024, 'max file size, MB bytes')
flags = tf.flags.FLAGS


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def sub_process(task_queue, order_queue, succ_queue, flags):
    print('sub Process start. process id: {} parent id: {}'.format(os.getpid(), os.getppid()))

    max_file_size = flags.max_file_size * 1024 * 1024
    output_path = flags.output_path
    image_folder = flags.image_folder

    file_order = order_queue.get()
    order_queue.put(file_order + 1)
    file_name = output_path + '.{:04}'.format(file_order)
    writer = tf.python_io.TFRecordWriter(file_name)

    curr_file_size = 0
    # normalize image to same size
    norm_width = flags.norm_width
    norm_height = flags.norm_height
    while True:
        ann = task_queue.get()
        if ann is None:
            print('Task None received, Ending...')
            writer.close()
            break
        image_path = os.path.join(image_folder, ann['filename'])
        if not os.path.exists(image_path):
            raise Exception('{} not found'.format(image_path))
        img = cv2.imread(image_path)
        shape = list(img.shape)
        img = cv2.resize(img, (norm_width, norm_height))
        img = img.astype(np.uint8)
        feature = {'train/label': _bytes_feature(tf.compat.as_bytes(json.dumps(ann))),
                   'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring())),
                   'train/shape': _int64_feature(shape)
                   }
        sample = tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()
        writer.write(sample)
        # check file size
        curr_file_size += len(sample)
        if curr_file_size > max_file_size:
            writer.close()
            file_order = order_queue.get()
            order_queue.put(file_order + 1)
            file_name = output_path + '.{:04}'.format(file_order)
            writer = tf.python_io.TFRecordWriter(file_name)
            curr_file_size = 0
        succ_queue.put(ann['filename'])


def parallel_write(argv):
    voc_labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                  "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    coco_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                   'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                   'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                   'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                   'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                   'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                   'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                   'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                   'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    labels = coco_labels
    annotations, _ = parse_voc_annotation(flags.anno_folder, '', flags.name_file, labels=labels)
    dir_name = os.path.dirname(flags.output_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    task_queue = Queue(flags.num_process * 2)
    # to keep file name order
    order_queue = Queue(1)
    # to keep successfully complete task
    succ_queue = Queue()  # must infinity

    process_pool = []
    total_succ = 0
    for k in range(0, flags.num_process):
        worker = Process(target=sub_process, args=(task_queue, order_queue, succ_queue, flags))
        worker.start()
        process_pool.append(worker)

    order_queue.put(0)
    for i, ann in enumerate(annotations):
        task_queue.put(ann)
        while not succ_queue.empty():
            succ_queue.get()
            total_succ += 1

    for i in range(len(process_pool)):
        task_queue.put(None)

    for worker in process_pool:
        worker.join()

    while not succ_queue.empty():
        succ_queue.get()
        total_succ += 1

    print('total annotations: {}, succ: {}'.format(len(annotations), total_succ))


if __name__ == "__main__":
    tf.app.run(parallel_write)
