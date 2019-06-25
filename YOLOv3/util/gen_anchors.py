import os
import json
import random
import argparse
import numpy as np
from xml.etree import ElementTree


def parse_voc_annotation(ann_dir, img_dir, name_file='', labels=[]):
    all_insts = []
    seen_labels = {}

    ann_lst = []
    if len(name_file) > 0 and os.path.exists(name_file):
        for name in open(name_file):
            ann_lst.append(name.strip() + '.xml')
    else:
        ann_lst = os.listdir(ann_dir)

    max_width, max_height, max_box_per_image = 0, 0, 0
    for ann in sorted(ann_lst):
        img = {'object': []}
        try:
            tree = ElementTree.parse(os.path.join(ann_dir, ann))
        except Exception as e:
            print(e)
            print('Ignore this bad annotation: ' + os.path.join(ann_dir, ann))
            continue
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = os.path.join(img_dir, elem.text)
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
                if img['width'] > max_width:
                    max_width = img['width']
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
                if img['height'] > max_height:
                    max_height = img['height']
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                addin, difficult = False, 0
                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text
                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1
                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]
                            addin = True
                    if 'difficult' in attr.tag:
                        difficult = int(attr.text)
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text))) - 1
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text))) - 1
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text))) - 1
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text))) - 1
                if addin and difficult > 0:
                    img['object'].pop()

        obj_num = len(img['object'])
        if obj_num > 0:
            all_insts += [img]
        if obj_num > max_box_per_image:
            max_box_per_image = obj_num
    print('max_width = {}, max_height = {}, max_box_per_image = {}'.format(
           max_width, max_height, max_box_per_image))
    return all_insts, seen_labels


def IOU(ann, centroids):
    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid

        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)  # will become (k,) shape

    return np.array(similarities)


def avg_IOU(anns, centroids):
    n, d = anns.shape
    sum = 0.
    for i in range(anns.shape[0]):
        sum += max(IOU(anns[i], centroids))
    return sum/n


def print_anchors(centroids, config):
    out_string = ''
    anchors = centroids.copy()
    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    net_h, net_w = config['model']['input_size'], config['model']['input_size']
    r = "anchors: ["
    for i in sorted_indices:
        out_string += str(int(anchors[i, 0] * net_w)) + ',' + str(int(anchors[i, 1] * net_h)) + ', '

    print(out_string[:-2])


def run_kmeans(ann_dims, anchor_num):
    ann_num = ann_dims.shape[0]
    iterations = 0
    prev_assignments = np.ones(ann_num) * (-1)
    iteration = 0
    old_distances = np.zeros((ann_num, anchor_num))

    indices = [random.randrange(ann_dims.shape[0]) for i in range(anchor_num)]
    centroids = ann_dims[indices]
    anchor_dim = ann_dims.shape[1]

    while True:
        distances = []
        iteration += 1
        for i in range(ann_num):
            d = 1 - IOU(ann_dims[i], centroids)
            distances.append(d)
        distances = np.array(distances)  # distances.shape = (ann_num, anchor_num)

        print("iteration {}: dists = {}".format(iteration, np.sum(np.abs(old_distances-distances))))

        # assign samples to centroids
        assignments = np.argmin(distances,axis=1)

        if (assignments == prev_assignments).all() :
            return centroids

        # calculate new centroids
        centroid_sums=np.zeros((anchor_num, anchor_dim), np.float)
        for i in range(ann_num):
            centroid_sums[assignments[i]] += ann_dims[i]
        for j in range(anchor_num):
            centroids[j] = centroid_sums[j] / (np.sum(assignments == j) + 1e-6)

        prev_assignments = assignments.copy()
        old_distances = distances.copy()


def main(args):
    conf_path = args.conf
    ann_dir = args.ann_dir
    img_dir = args.img_dir
    name_file = args.name_file
    num_anchor = args.num_anchor

    with open(conf_path) as conf_buffer:
        config = json.loads(conf_buffer.read())
    train_imgs, train_labels = parse_voc_annotation(ann_dir, img_dir, name_file, config['model']['labels'])

    # run k_mean to find the anchors
    annotation_dims = []
    for image in train_imgs:
        # print(image['filename'])
        try:
            for obj in image['object']:
                relative_w = (float(obj['xmax']) - float(obj['xmin'])) / image['width']
                relatice_h = (float(obj["ymax"]) - float(obj['ymin'])) / image['height']
                annotation_dims.append(tuple(map(float, (relative_w,relatice_h))))
        except Exception as e:
            print(image)
            print(image['filename'])
            print(image['width'], image['height'])
            print(float(obj['xmax']), float(obj['xmin']), float(obj["ymax"]), float(obj['ymin']))
            raise e

    annotation_dims = np.array(annotation_dims)
    centroids = run_kmeans(annotation_dims, num_anchor)

    # write anchors to file
    print('\naverage IOU for', num_anchor, 'anchors:', '%0.2f' % avg_IOU(annotation_dims, centroids))
    print_anchors(centroids, config)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '-c',
        '--conf',
        default='config.json',
        help='path to configuration file')
    arg_parser.add_argument(
        '-a',
        '--ann_dir',
        default='',
        help='path to annotation directory')
    arg_parser.add_argument(
        '-i',
        '--img_dir',
        default='',
        help='path to image directory')
    arg_parser.add_argument(
        '-f',
        '--name_file',
        default='',
        help='path to image name file')
    arg_parser.add_argument(
        '-n',
        '--num_anchor',
        default=9,
        help='number of anchors to use')

    args = arg_parser.parse_args()
    main(args)
