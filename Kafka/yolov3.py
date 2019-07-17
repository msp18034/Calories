import numpy as np
from keras.models import load_model, model_from_json
import cv2
import time


class YOLO:
    def __init__(self):
        """Init.

        # Arguments
            obj_threshold: Integer, threshold for object.
            nms_threshold: Integer, threshold for box.
        """
        self._t1 = 0.25  # obj_threshold
        self._t2 = 0.45  # nms_threshold
        model_path = '/home/hduser/model_weights/yolo.h5'
        #model_path = 'D:/0study/Project/code/YOLOv3-master-xian/data/yolo.h5'
        model = load_model(model_path)
        self.model_dic = self.serialize_keras_model(model)
        classes_path = '/home/hduser/Calories/dataset/coco_classes.txt'
        #classes_path = '../dataset/coco_classes.txt'
        self.class_name = self._get_class(classes_path)

    def _get_class(self, path):
        with open(path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def deserialize_keras_model(self, dictionary):
        """Deserialized the Keras model using the specified dictionary."""
        architecture = dictionary['model']
        weights = dictionary['weights']
        model = model_from_json(architecture)
        model.set_weights(weights)
        return model

    def serialize_keras_model(self, model):
        """Serializes the specified Keras model into a dictionary."""
        dictionary = {}
        dictionary['model'] = model.to_json()
        dictionary['weights'] = model.get_weights()
        return dictionary

    def _sigmoid(self, x):
        """sigmoid.

        # Arguments
            x: Tensor.

        # Returns
            numpy ndarray.
        """
        return 1 / (1 + np.exp(-x))

    def _process_feats(self, out, anchors, mask):
        """process output features.

        # Arguments
            out: Tensor (N, N, 3, 4 + 1 +80), output feature map of yolo.
            anchors: List, anchors for box.
            mask: List, mask for anchors.

        # Returns
            boxes: ndarray (N, N, 3, 4), x,y,w,h for per box.
            box_confidence: ndarray (N, N, 3, 1), confidence for per box.
            box_class_probs: ndarray (N, N, 3, 80), class probs for per box.
        """
        grid_h, grid_w, num_boxes = map(int, out.shape[1: 4])

        anchors = [anchors[i] for i in mask]
        anchors_tensor = np.array(anchors).reshape(1, 1, len(anchors), 2)

        # Reshape to batch, height, width, num_anchors, box_params.
        out = out[0]
        box_xy = self._sigmoid(out[..., :2])
        box_wh = np.exp(out[..., 2:4])
        box_wh = box_wh * anchors_tensor

        box_confidence = self._sigmoid(out[..., 4])
        box_confidence = np.expand_dims(box_confidence, axis=-1)
        box_class_probs = self._sigmoid(out[..., 5:])

        col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

        col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)

        box_xy += grid
        box_xy /= (grid_w, grid_h)
        box_wh /= (416, 416)
        box_xy -= (box_wh / 2.)
        boxes = np.concatenate((box_xy, box_wh), axis=-1)

        return boxes, box_confidence, box_class_probs

    def _filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes with object threshold.

        # Arguments
            boxes: ndarray, boxes of objects.
            box_confidences: ndarray, confidences of objects.
            box_class_probs: ndarray, class_probs of objects.

        # Returns
            boxes: ndarray, filtered boxes.
            classes: ndarray, classes for boxes.
            scores: ndarray, scores for boxes.
        """
        box_scores = box_confidences * box_class_probs
        box_classes = np.argmax(box_scores, axis=-1)
        box_class_scores = np.max(box_scores, axis=-1)
        pos = np.where(box_class_scores >= self._t1)

        boxes = boxes[pos]
        classes = box_classes[pos]
        scores = box_class_scores[pos]

        return boxes, classes, scores

    def _nms_boxes(self, boxes, scores):
        """Suppress non-maximal boxes.

        # Arguments
            boxes: ndarray, boxes of objects.
            scores: ndarray, scores of objects.

        # Returns
            keep: ndarray, index of effective boxes.
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 1)
            h1 = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self._t2)[0]
            order = order[inds + 1]

        keep = np.array(keep)

        return keep

    def _yolo_out(self, outs, shape):
        """Process output of yolo base net.

        # Argument:
            outs: output of yolo base net.
            shape: shape of original image.

        # Returns:
            boxes: ndarray, boxes of objects.
            classes: ndarray, classes of objects.
            scores: ndarray, scores of objects.
        """
        masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                   [59, 119], [116, 90], [156, 198], [373, 326]]

        boxes, classes, scores = [], [], []

        for out, mask in zip(outs, masks):
            b, c, s = self._process_feats(out, anchors, mask)
            b, c, s = self._filter_boxes(b, c, s)
            boxes.append(b)
            classes.append(c)
            scores.append(s)

        boxes = np.concatenate(boxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)

        # Scale boxes back to original image shape.
        width, height = shape[1], shape[0]
        image_dims = [width, height, width, height]
        boxes = boxes * image_dims

        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]

            keep = self._nms_boxes(b, s)

            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return boxes, classes, scores

    def predict(self, image, shape):
        """Detect the objects with yolo.

        # Arguments
            image: ndarray, processed input image.
            shape: shape of original image.

        # Returns
            boxes: ndarray, boxes of objects.
            classes: ndarray, classes of objects.
            scores: ndarray, scores of objects.
        """
        model = self.deserialize_keras_model(self.model_dic)
        outs = model.predict(image)
        boxes, classes, scores = self._yolo_out(outs, shape)

        return boxes, classes, scores

    def process_image(self, img):
        """Resize, reduce and expand image.

        # Argument:
            img: original PIL image.

        # Returns
            image: ndarray(1, 416, 416, 3), processed image.
        """
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        image = cv2.resize(img, (416, 416),
                           interpolation=cv2.INTER_CUBIC)
        image = np.array(image, dtype='float32')
        image /= 255.
        image = np.expand_dims(image, axis=0)

        return image


    def fliter(self, image, boxes, scores, classes):
        """找到碗和勺子的照片，碗的框
        """
        bowl_box = []
        food_img = []
        spoon_score = 0
        spoon_img = 0
        for box, score, cl in zip(boxes, scores, classes):
            x, y, w, h = box

            left = max(0, np.floor(x + 0.5).astype(int))
            top = max(0, np.floor(y + 0.5).astype(int))
            right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
            bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))
            box = (left, top, right, bottom)

            if self.class_name[cl] == 'bowl':
                cropped_img = image.crop(box)
                cropped_img = cv2.cvtColor(np.asarray(cropped_img), cv2.COLOR_RGB2BGR)
                food_img.append(cropped_img)
                bowl_box.append(box)
            elif self.class_name[cl] == 'spoon':
                if score > spoon_score:
                    spoon_img = image.crop(box)
                    spoon_score = score
        return spoon_img, bowl_box, food_img



    def detect_food(self, image):
        """Use yolo v3 to detect images.

        # Argument:
            image: original image.
            yolo: YOLO, yolo model.
            all_classes: all classes name.

        # Returns:
            boxes of foods and spoon
        """
        pimage = self.process_image(image)

        start = time.time()
        boxes, classes, scores = self.predict(pimage, image.shape)
        spoon_img, bowl_box, food_img = self.fliter(image, boxes, scores, classes)
        end = time.time()
        print('Yolo time: {0:.2f}s'.format(end - start))

        return bowl_box, food_img, spoon_img
