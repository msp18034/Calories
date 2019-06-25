#! /usr/bin/env python

import os
import cv2
import json
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from xml.etree import ElementTree
from util.utils import preprocess_input
from util.evaluator import Evaluator

MODEL_FLOPS = None


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name='',
            op_dict=None,
            producer_op_list=None
        )
    return graph


def load_model(args):
    with open(args.config) as config_buffer:
        config = json.load(config_buffer)
    graph = load_graph(args.model)

    if config['model']['architecture'] in ('YOLOFLY',):
        input_image = graph.get_tensor_by_name('input_image:0')
        layer_29 = graph.get_tensor_by_name('layer_29/BiasAdd:0')
        layer_25 = graph.get_tensor_by_name('layer_25/BiasAdd:0')
        layer_21 = graph.get_tensor_by_name('layer_21/BiasAdd:0')
        output_nodes = [layer_29, layer_25, layer_21]
    elif config['model']['architecture'] in ('DarkNet53',):
        input_image = graph.get_tensor_by_name('input_image:0')
        conv_81 = graph.get_tensor_by_name('conv_81/BiasAdd:0')
        conv_93 = graph.get_tensor_by_name('conv_93/BiasAdd:0')
        conv_105 = graph.get_tensor_by_name('conv_105/BiasAdd:0')
        output_nodes = [conv_81, conv_93, conv_105]
    else:
        raise Exception('unknow model architecture:{}'.format(config['model']['architecture']))

    sess = tf.Session(graph=graph)
    return sess, input_image, output_nodes, config


def load_image(args):
    input_path = args.input
    name_file = args.name_file
    image_paths = []
    if name_file is not None and os.path.exists(name_file):
        for name in open(name_file):
            for ext in ['.jpg', '.png', '.jpeg']:
                image_path = os.path.join(input_path, name.strip() + ext)
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                    break
    else:
        if os.path.isdir(input_path):
            for inp_file in os.listdir(input_path):
                image_paths += [os.path.join(input_path, inp_file)]
        else:
            image_paths += [input_path]

        image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]
    print('load image count = {}'.format(len(image_paths)))
    return image_paths


def load_label(args):
    label_path = args.annotation
    name_file = args.name_file
    annotation_paths = []
    if name_file is not None and os.path.exists(name_file):
        for name in open(name_file):
            annotation_paths.append(os.path.join(label_path, name.strip() + '.xml'))
    else:
        if os.path.isdir(label_path):
            for inp_file in os.listdir(label_path):
                annotation_paths += [os.path.join(label_path, inp_file)]
        else:
            annotation_paths += [label_path]

    ann_cnt, obj_cnt = 0, 0

    labels = {}
    with open(args.config) as config_buffer:
        config = json.load(config_buffer)
    for f in annotation_paths:
        if not f.endswith('.xml'):
            continue
        basename = os.path.basename(f)
        basename = os.path.splitext(basename)[0]
        tree = ElementTree.parse(open(f))

        img = {'object': []}
        for elem in tree.iter():
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                add_flag = False
                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text
                        if obj['name'] not in config['model']['labels']:
                            add_flag = False
                            break
                        else:
                            add_flag = True
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))
                    if 'difficult' in attr.tag:
                        if int(attr.text) > 0:
                            add_flag = False
                            break
                if add_flag:
                    img['object'] += [obj]

        if len(img['object']) > 0:
            labels[basename] = json.dumps(img)
            ann_cnt += 1
            obj_cnt += len(img['object'])

    print('load annotation count = {}, object count = {}'.format(ann_cnt, obj_cnt))
    return labels


def inference(sess, input_image, output_nodes, images, net_h, net_w):
    global MODEL_FLOPS
    nb_images = len(images)
    batch_input = np.zeros((nb_images, net_h, net_w, 3))
    for i in range(nb_images):
        batch_input[i] = preprocess_input(images[i], net_h, net_w)
    batch_output = sess.run(output_nodes, feed_dict={
        input_image: batch_input
    })

    if MODEL_FLOPS is None:
        run_metadata = tf.RunMetadata()
        sess.run(output_nodes,
                 options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                 run_metadata=run_metadata,
                 feed_dict={input_image: batch_input})
        # get model flops
        flops = tf.profiler.profile(sess.graph, cmd='op', run_meta=run_metadata,
                                    options=tf.profiler.ProfileOptionBuilder.float_operation())
        MODEL_FLOPS = flops.total_float_ops / nb_images

    return batch_output, batch_input


def evaluate(args):
    global MODEL_FLOPS
    image_paths = load_image(args)
    annotations = load_label(args)

    sess, input_image, output_nodes, config = load_model(args)
    evaluator = Evaluator(config=config, augment=None, params={
        'obj_thresh': 1e-2, 'nms_thresh': 0.45,
        'images': None, 'annotations': None, 'shapes': None, 'loss': None, 'output_nodes': None
    })

    net_h, net_w = config['model']['input_size'], config['model']['input_size']

    all_gts = []
    all_tps = []
    all_fps = []
    all_scores = []

    num_class = len(config['model']['labels'])
    for i in range(num_class):
        all_gts.append(0)
        all_tps.append([])
        all_fps.append([])
        all_scores.append([])

    total_image_cnt, batch_size, idx = len(image_paths), 40, 0
    images, shapes, annos = [], [], []
    for image_path in tqdm(image_paths):
        idx += 1

        basename = os.path.splitext(os.path.basename(image_path))[0]
        if basename not in annotations:
            print('no found {}'.format(basename))
            continue
        anno = np.array(annotations[basename])
        img = cv2.imread(image_path)

        images.append(img)
        shapes.append(img.shape)
        annos.append(anno)

        if idx % batch_size == 0 or idx == total_image_cnt:
            batch_output, batch_input = inference(sess, input_image, output_nodes, images, net_h, net_w)
            batch_boxes = evaluator.get_boxes(np.asarray(shapes), batch_input, batch_output, coco=False,
                                              nms='nms', cls='softmax')
            labels = evaluator.get_labels(np.asarray(annos))
            tps, fps, scores, gts, _, _ = evaluator.get_tp_fp_case(labels, batch_boxes)

            for i in range(num_class):
                all_gts[i] += gts[i]
                all_tps[i] += tps[i]
                all_fps[i] += fps[i]
                all_scores[i] += scores[i]

            images, shapes, annos = [], [], []

    all_ap = evaluator.evaluate(all_gts, all_tps, all_fps, all_scores)
    m_ap, num = 0, 0
    for i, ap in enumerate(all_ap):
        print('{:15} AP = {:.6f}'.format(config['model']['labels'][i], ap))
        if ap > -1:
            m_ap += ap
            num += 1
    if num > 0:
        m_ap /= float(num)
    print('mAP = {}'.format(m_ap))
    if MODEL_FLOPS is not None:
        print('Model FLOPS = {:.2f} Bn'.format(MODEL_FLOPS / 1e+9))


def predict(args):
    image_paths = load_image(args)

    sess, input_image, output_nodes, config = load_model(args)
    evaluator = Evaluator(config=config, augment=None, params={
        'obj_thresh': 1e-2, 'nms_thresh': 0.45,
        'images': None, 'annotations': None, 'shapes': None, 'loss': None, 'output_nodes': None
    })

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_lst = []
    for name in config['model']['labels']:
        f = os.path.join(output_dir, 'comp3_det_test_' + name + '.txt')
        out = open(f, 'w')
        out_lst.append(out)

    net_h, net_w = config['model']['input_size'], config['model']['input_size']

    total_image_cnt, batch_size, idx = len(image_paths), 20, 0
    images, shapes, names = [], [], []
    for image_path in tqdm(image_paths):
        idx += 1

        img = cv2.imread(image_path)
        images.append(img)
        shapes.append(img.shape)

        basename = os.path.splitext(os.path.basename(image_path))[0]
        names.append(basename)

        if idx % batch_size == 0 or idx == total_image_cnt:
            batch_output, batch_input = inference(sess, input_image, output_nodes, images, net_h, net_w)
            shapes = np.asarray(shapes)
            batch_boxes = evaluator.get_boxes(shapes, batch_input, batch_output, coco=True,
                                              nms='nms', cls='softmax')

            for k, class_boxes in enumerate(batch_boxes):
                for i, boxes in enumerate(class_boxes):
                    if len(boxes) == 0:
                        continue
                    for box in boxes:
                        left, top, right, bottom, score = box[:5]
                        out_lst[i].write('{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                            names[k], score, left + 1, top + 1, right + 1, bottom + 1))
            images, shapes, names = [], [], []

    for out in out_lst:
        out.close()


def predict_coco(args):
    image_paths = load_image(args)

    sess, input_image, output_nodes, config = load_model(args)
    evaluator = Evaluator(config=config, augment=None, params={
        'obj_thresh': 1e-2, 'nms_thresh': 0.45,
        'images': None, 'annotations': None, 'shapes': None, 'loss': None, 'output_nodes': None
    })

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = 'detections_test-dev2017_yolov3_results.json'
    output_path = os.path.join(output_dir, output_path)
    out = open(output_path, 'w')

    cat_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    cat_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31,
               32, 33, 34, 35,36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
               58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87,
               88, 89, 90]
    cat_maps = dict(zip(cat_names, cat_ids))

    out_lst = []
    net_h, net_w = config['model']['input_size'], config['model']['input_size']

    total_image_cnt, batch_size, idx = len(image_paths), 20, 0
    images, shapes, names = [], [], []
    for image_path in tqdm(image_paths):
        idx += 1

        img = cv2.imread(image_path)
        images.append(img)
        shapes.append(img.shape)

        basename = os.path.splitext(os.path.basename(image_path))[0]
        img_id = int(basename.split('_')[-1])
        names.append(img_id)

        if idx % batch_size == 0 or idx == total_image_cnt:
            batch_output, batch_input = inference(sess, input_image, output_nodes, images, net_h, net_w)
            shapes = np.asarray(shapes)
            batch_boxes = evaluator.get_boxes(shapes, batch_input, batch_output, coco=True,
                                              nms='nms', cls='softmax')

            for k, class_boxes in enumerate(batch_boxes):
                for i, boxes in enumerate(class_boxes):
                    if len(boxes) == 0:
                        continue
                    category_id = cat_maps[config['model']['labels'][i]]
                    for box in boxes:
                        left, top, right, bottom, score = box[:5]
                        x = round(left, 4)
                        y = round(top, 4)
                        width = round(right - left, 4)
                        height = round(bottom - top, 4)
                        score = round(float(score), 6)
                        out_lst.append({'image_id': names[k], 'category_id': category_id,
                                        'bbox': [x, y, width, height], 'score': score})
            images, shapes, names = [], [], []

    out.write(json.dumps(out_lst))
    out.close()


def main(args):
    if args.annotation is not None:
        evaluate(args)
    elif args.type == 'voc':
        predict(args)
    elif args.type == 'coco':
        predict_coco(args)
    else:
        raise Exception('unknow type:{}'.format(args.type))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--input')
    arg_parser.add_argument('-a', '--annotation', default=None)
    arg_parser.add_argument('-f', '--name_file', default=None)
    arg_parser.add_argument('-o', '--output_dir', default=None)
    arg_parser.add_argument('-m', '--model')
    arg_parser.add_argument('-c', '--config')
    arg_parser.add_argument('-t', '--type', default='voc')

    args = arg_parser.parse_args()
    main(args)
