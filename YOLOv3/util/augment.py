import cv2
import json
import numpy as np
from utils import BoundBox, bbox_iou
from image import apply_random_scale_and_crop, random_distort_image, random_flip, correct_bounding_boxes


class Augment(object):

    def __init__(self, config):
        self.config = config
        self.batch_size = config['train']['batch_size']
        self.labels = config['model']['labels']
        self.down_sample = 32
        self.max_box_per_image = config['train']['max_box_per_image']
        self.min_net_size = (config['model']['min_input_size'] // self.down_sample) * self.down_sample
        self.max_net_size = (config['model']['max_input_size'] // self.down_sample) * self.down_sample
        self.jitter = 0.3
        self.norm = lambda t: t / 255.0
        self.anchors = [BoundBox(0, 0, config['model']['anchors'][2 * i], config['model']['anchors'][2 * i + 1])
                        for i in range(len(config['model']['anchors']) // 2)]
        self.net_h = config['model']['input_size']
        self.net_w = config['model']['input_size']

        self.idx = 0

    def __call__(self, images, annotations, shapes, aug=True):
        # get image input size, change every 10 batches
        if aug:
            self.idx += 1
            net_h, net_w = self._get_net_size()
        else:
            net_h, net_w = self.config['model']['input_size'], self.config['model']['input_size']

        base_grid_h, base_grid_w = net_h // self.down_sample, net_w // self.down_sample

        x_batch = np.zeros((self.batch_size, net_h, net_w, 3), dtype=np.float32)
        t_batch = np.zeros((self.batch_size, 1, 1, 1, self.max_box_per_image, 4), dtype=np.float32)

        # initialize the inputs and the outputs
        yolo_1 = np.zeros((self.batch_size, 1 * base_grid_h, 1 * base_grid_w, len(self.anchors) // 3, 4 + 1 + len(self.labels)), dtype=np.float32)
        yolo_2 = np.zeros((self.batch_size, 2 * base_grid_h, 2 * base_grid_w, len(self.anchors) // 3, 4 + 1 + len(self.labels)), dtype=np.float32)
        yolo_3 = np.zeros((self.batch_size, 4 * base_grid_h, 4 * base_grid_w, len(self.anchors) // 3, 4 + 1 + len(self.labels)), dtype=np.float32)
        yolos = [yolo_3, yolo_2, yolo_1]

        instance_count = 0
        true_box_index = 0

        # do the logic to fill in the inputs and the output
        for img, ann, shape in zip(images, annotations, shapes):
            ann = json.loads(ann)
            img = cv2.resize(img, (shape[1], shape[0]))
            # augment input image and fix object's position and size
            if aug:
                img, all_objs = self._aug_image(img, ann, net_h, net_w)
            else:
                img, all_objs = self._raw_image(img, ann, net_h, net_w)

            for obj in all_objs:
                # find the best anchor box for this object
                max_anchor = None
                max_index = -1
                max_iou = -1
                # not only max iou anchor but also larger than threshold anchors are positive.
                positive_anchors = []
                positive_threshold = 0.3

                shifted_box = BoundBox(0, 0, obj['xmax'] - obj['xmin'], obj['ymax'] - obj['ymin'])

                for i in range(len(self.anchors)):
                    anchor = self.anchors[i]
                    iou = bbox_iou(shifted_box, anchor)

                    if max_iou < iou:
                        max_anchor = anchor
                        max_index = i
                        max_iou = iou
                    if iou > positive_threshold:
                        positive_anchors.append([i, anchor])
                if not positive_anchors:
                    positive_anchors.append([max_index, max_anchor])

                for max_index, max_anchor in positive_anchors:
                    # determine the yolo to be responsible for this bounding box
                    yolo = yolos[max_index // 3]
                    grid_h, grid_w = yolo.shape[1:3]

                    # determine the position of the bounding box on the grid
                    center_x = .5*(obj['xmin'] + obj['xmax'])
                    center_x = center_x / float(net_w) * grid_w  # sigma(t_x) + c_x
                    center_y = .5*(obj['ymin'] + obj['ymax'])
                    center_y = center_y / float(net_h) * grid_h  # sigma(t_y) + c_y

                    # determine the sizes of the bounding box
                    w = np.log((obj['xmax'] - obj['xmin']) / float(max_anchor.xmax))  # t_w
                    h = np.log((obj['ymax'] - obj['ymin']) / float(max_anchor.ymax))  # t_h

                    box = [center_x, center_y, w, h]

                    # determine the index of the label
                    obj_indx = self.labels.index(obj['name'])

                    # determine the location of the cell responsible for this object
                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    # assign ground truth x, y, w, h, confidence and class probs to y_batch
                    yolo[instance_count, grid_y, grid_x, max_index % 3] = 0
                    yolo[instance_count, grid_y, grid_x, max_index % 3, 0:4] = box
                    yolo[instance_count, grid_y, grid_x, max_index % 3, 4] = 1.
                    yolo[instance_count, grid_y, grid_x, max_index % 3, 5+obj_indx] = 1

                    # assign the true box to t_batch
                    true_box = [center_x, center_y, obj['xmax'] - obj['xmin'], obj['ymax'] - obj['ymin']]
                    t_batch[instance_count, 0, 0, 0, true_box_index] = true_box

                    true_box_index += 1
                    true_box_index = true_box_index % self.max_box_per_image

            # assign input image to x_batch
            if aug and self.norm is not None:
                x_batch[instance_count] = self.norm(img)
            elif not aug:
                x_batch[instance_count] = img
            # increase instance counter in the current batch
            instance_count += 1

        output = [x_batch, t_batch, yolo_1, yolo_2, yolo_3]
        if not aug:
            output += [images, annotations, shapes]
        return output

    def _get_net_size(self):
        if self.idx % 10 == 0:
            net_size = self.down_sample * np.random.randint(self.min_net_size / self.down_sample,
                                                            self.max_net_size / self.down_sample + 1)
            self.net_h, self.net_w = net_size, net_size
        return self.net_h, self.net_w

    def _aug_image(self, image, annotation, net_h, net_w):
        image = image[:, :, ::-1]  # RGB image
        image_h, image_w, _ = image.shape

        # determine the amount of scaling and cropping
        dw = self.jitter * image_w
        dh = self.jitter * image_h

        new_ar = (image_w + np.random.uniform(-dw, dw)) / (image_h + np.random.uniform(-dh, dh))
        scale = np.random.uniform(0.25, 2)
        # scale = 1.0

        if new_ar < 1:
            new_h = int(scale * net_h)
            new_w = int(net_h * new_ar)
        else:
            new_w = int(scale * net_w)
            new_h = int(net_w / new_ar)

        dx = int(np.random.uniform(0, net_w - new_w))
        dy = int(np.random.uniform(0, net_h - new_h))

        # apply scaling and cropping
        im_sized = apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy)

        # randomly distort hsv space
        im_sized = random_distort_image(im_sized)

        # randomly flip
        flip = np.random.randint(2)
        im_sized = random_flip(im_sized, flip)

        # correct the size and pos of bounding boxes
        all_objs = correct_bounding_boxes(annotation['object'], new_w, new_h, net_w, net_h,
                                          dx, dy, flip, image_w, image_h)

        return im_sized, all_objs

    def _raw_image(self, image, annotation, net_h, net_w):
        image_h, image_w, _ = image.shape
        # determine the new size of the image
        if (float(net_w) / image_w) < (float(net_h) / image_h):
            new_h = (image_h * net_w) // image_w
            new_w = net_w
        else:
            new_w = (image_w * net_h) // image_h
            new_h = net_h

        # resize the image to the new size
        resized = cv2.resize(image[:, :, ::-1], (new_w, new_h)) / 255.0

        # embed the image into the standard letter box
        new_image = np.ones((net_h, net_w, 3)) * 0.5
        new_image[(net_h-new_h)//2:(net_h+new_h)//2, (net_w-new_w)//2:(net_w+new_w)//2, :] = resized
        dx, dy = (net_w - new_w) // 2, (net_h - new_h) // 2
        all_objs = correct_bounding_boxes(annotation['object'], new_w, new_h, net_w, net_h,
                                          dx, dy, 0, image_w, image_h)
        return new_image, all_objs
