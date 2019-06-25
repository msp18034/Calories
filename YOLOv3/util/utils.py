import os
import cv2
import numpy as np
from scipy.special import expit


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def _sigmoid(x):
    return expit(x)


def _softmax(x, axis=-1):
    x = x - np.amax(x, axis, keepdims=True)
    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin

    union = w1*h1 + w2*h2 - intersect
    if union > 0:
        return float(intersect) / union
    return 0


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w, coco=False):
    if (float(net_w)/image_w) < (float(net_h)/image_h):
        new_w = net_w
        new_h = (image_h*net_w)/image_w
    else:
        new_h = net_w
        new_w = (image_w*net_h)/image_h

    invalid_ids = []
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h

        if coco:
            boxes[i].xmin = max(0., (boxes[i].xmin - x_offset) / x_scale * image_w)
            boxes[i].xmax = min(image_w - 1., (boxes[i].xmax - x_offset) / x_scale * image_w)
            boxes[i].ymin = max(0., (boxes[i].ymin - y_offset) / y_scale * image_h)
            boxes[i].ymax = min(image_h - 1., (boxes[i].ymax - y_offset) / y_scale * image_h)
        else:
            boxes[i].xmin = max(0, int((boxes[i].xmin - x_offset) / x_scale * image_w))
            boxes[i].xmax = min(int(image_w) - 1, int((boxes[i].xmax - x_offset) / x_scale * image_w))
            boxes[i].ymin = max(0, int((boxes[i].ymin - y_offset) / y_scale * image_h))
            boxes[i].ymax = min(int(image_h) - 1, int((boxes[i].ymax - y_offset) / y_scale * image_h))

        if boxes[i].xmin >= boxes[i].xmax or boxes[i].ymin >= boxes[i].ymax:
            invalid_ids.append(i)
    for i in invalid_ids[::-1]:
        boxes.pop(i)


def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return

    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue

            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0


def do_mat_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return

    x1 = np.array([box.xmin for box in boxes])
    y1 = np.array([box.ymin for box in boxes])
    x2 = np.array([box.xmax for box in boxes])
    y2 = np.array([box.ymax for box in boxes])

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for c in range(nb_class):
        scores = np.array([box.classes[c] for box in boxes])
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / np.maximum((areas[i] + areas[order[1:]] - inter), np.finfo(np.float32).eps)

            idx = np.where(ovr < nms_thresh)[0]
            order = order[idx + 1]
        keep.sort()
        for j in range(len(boxes)):
            if keep and j == keep[0]:
                keep.pop(0)
                continue
            if boxes[j].classes[c] > 0:
                boxes[j].classes[c] = 0


def do_cv_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return

    new_boxes = [list(map(float, [b.xmin, b.ymin, b.xmax - b.xmin + 1, b.ymax - b.ymin + 1])) for b in boxes]
    for c in range(nb_class):
        scores = [float(b.classes[c]) for b in boxes]
        keep = cv2.dnn.NMSBoxes(bboxes=new_boxes, scores=scores, score_threshold=0.0,
                                nms_threshold=nms_thresh, top_k=100)
        keep = [t[0] for t in keep]
        keep.sort()
        for j in range(len(boxes)):
            if keep and j == keep[0]:
                keep.pop(0)
                continue
            if boxes[j].classes[c] > 0:
                boxes[j].classes[c] = 0


def do_fast_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return

    x1 = np.array([box.xmin for box in boxes])
    y1 = np.array([box.ymin for box in boxes])
    x2 = np.array([box.xmax for box in boxes])
    y2 = np.array([box.ymax for box in boxes])

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    scores = np.array([box.c for box in boxes])
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        ovr = inter / np.maximum((areas[i] + areas[order[1:]] - inter), np.finfo(np.float32).eps)

        idx = np.where(ovr < nms_thresh)[0]
        order = order[idx + 1]
    keep.sort()
    for j in range(len(boxes)):
        if keep and j == keep[0]:
            keep.pop(0)
            continue
        boxes[j].classes = np.zeros(nb_class)
        boxes[j].c = 0


def decode_netout(netout, anchors, obj_thresh, net_h, net_w, cls='sigmoid'):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5

    boxes = []

    netout[..., :2] = _sigmoid(netout[..., :2])

    if cls == 'sigmoid':
        netout[..., 4:] = _sigmoid(netout[..., 4:])
    elif cls == 'softmax':
        netout[..., 4] = _sigmoid(netout[..., 4])
        netout[..., 5:] = _softmax(netout[..., 5:])
    else:
        raise Exception('unknow cls function:{}'.format(cls))

    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h*grid_w):
        row = i // grid_w
        col = i % grid_w

        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[row, col, b, 4]

            if(objectness <= obj_thresh): continue

            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[row,col,b,:4]

            x = (col + x) / grid_w  # center position, unit: image width
            y = (row + y) / grid_h  # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height

            # last elements are class probabilities
            classes = netout[row,col,b,5:]

            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)

            boxes.append(box)

    return boxes


def decode_netout_mat(netout, anchors, obj_thresh, net_h, net_w, cls='sigmoid'):
    anchors = np.asarray(anchors)
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5

    cell_x = np.reshape(np.tile(range(grid_w), [grid_h]), (grid_h, grid_w, 1, 1))
    cell_y = np.transpose(cell_x, (1, 0, 2, 3))
    cell_grid = np.tile(np.concatenate([cell_x, cell_y], -1), [1, 1, nb_box, 1])

    xy = (_sigmoid(netout[..., :2]) + cell_grid) / np.reshape([grid_w, grid_h], [1, 1, 1, 2])
    wh = np.exp(netout[..., 2:4]) * np.reshape(anchors, [1, 1, nb_box, 2]) / np.reshape([net_w, net_h], [1, 1, 1, 2])
    conf = _sigmoid(netout[..., 4:5])

    if cls == 'sigmoid':
        prob = conf * _sigmoid(netout[..., 5:])
    elif cls == 'softmax':
        prob = conf * _softmax(netout[..., 5:])
    else:
        raise Exception('unknow cls function:{}'.format(cls))

    prob *= prob > obj_thresh

    xy_min = np.maximum(xy - wh / 2.0, 0.0)
    xy_max = np.minimum(xy + wh / 2.0, 1.0)

    valid_index = np.where(np.squeeze(conf, axis=-1) > obj_thresh)
    xy_min_valid = xy_min[valid_index]
    xy_max_valid = xy_max[valid_index]
    conf_valid = conf[valid_index]
    prob = prob[valid_index]

    boxes = []
    for i in range(len(xy_min_valid)):
        x_min, y_min = xy_min_valid[i]
        x_max, y_max = xy_max_valid[i]
        objectness = conf_valid[i][0]
        classes = prob[i]
        boxes.append(BoundBox(x_min, y_min, x_max, y_max, objectness, classes))

    return boxes


def preprocess_input(image, net_h, net_w):
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(net_w)/new_w) < (float(net_h)/new_h):
        new_h = (new_h * net_w)//new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h)//new_h
        new_h = net_h

    # resize the image to the new size
    resized = cv2.resize(image, (new_w, new_h))[:,:,::-1] / 255.0

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[(net_h-new_h)//2:(net_h+new_h)//2, (net_w-new_w)//2:(net_w+new_w)//2, :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image


