import cv2
import json
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from utils import decode_netout_mat, correct_yolo_boxes, do_nms, do_mat_nms, do_fast_nms, do_cv_nms

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color='red', thickness=4, display_str_list=(),
                               use_normalized_coordinates=True):
    """Adds a bounding box to an image.
    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.
    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.
    Args:
        image: a PIL.Image object.
        ymin: ymin of bounding box.
        xmin: xmin of bounding box.
        ymax: ymax of bounding box.
        xmax: xmax of bounding box.
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default value is 4.
        display_str_list: list of strings to display in box
                          (each to be shown on its own line).
        use_normalized_coordinates: If True (default), treat coordinates
          ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
          coordinates as absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        # text_bottom = bottom + total_display_str_height
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                              text_bottom)],
            fill=color)
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font)
        text_bottom -= text_height - 2 * margin


class Evaluator(object):

    def __init__(self, config, augment, params):
        self.config = config
        self.augment = augment

        self.val_data = params['data'] if 'data' in params else None
        self.val_loss = params['loss'] if 'loss' in params else None
        self.output_nodes = params['output_nodes']

        self.obj_thresh = params['obj_thresh'] if 'obj_thresh' in params else 0.25
        self.nms_thresh = params['nms_thresh'] if 'nms_thresh' in params else 0.45

        self.nms_func = params['nms_func'] if 'nms_func' in params else 'fast_nms'
        self.cls_func = params['cls_func'] if 'cls_func' in params else 'sigmoid'

    def run(self, sess):
        total_loss = 0
        total_step = 0

        all_gts = []
        all_tps = []
        all_fps = []
        all_scores = []

        num_class = len(self.config['model']['labels'])
        for i in range(num_class):
            all_gts.append(0)
            all_tps.append([])
            all_fps.append([])
            all_scores.append([])

        # for summary show case
        worst_cnt = -np.Inf
        worst_img = None

        for k in range(self.config['val']['epoch_size']):
            if k % 10 == 0:
                print('Evaluate Batch {}'.format(k))
            v_input = sess.run(self.val_data)
            outputs = sess.run([self.val_loss] + self.output_nodes,
                               feed_dict={'input_image:0': v_input[0],
                                          'true_boxes:0': v_input[1],
                                          'true_yolo_1:0': v_input[2],
                                          'true_yolo_2:0': v_input[3],
                                          'true_yolo_3:0': v_input[4],
                                          'phase:0': False,
                                          # 'conv1_bn/keras_learning_phase:0': False,
                                          }
                               )
            loss = outputs[0]
            total_loss += loss
            total_step += 1
            v_images, v_annotations, v_shapes = v_input[5:8]
            predicts = self.get_boxes(v_shapes, v_input[0], outputs[1:], nms=self.nms_func, cls=self.cls_func)
            labels = self.get_labels(v_annotations)
            # tp, fp, score, gt_num = self.get_tp_fp(labels, predicts)
            tps, fps, scores, gts, bad_cnt, bad_idx = self.get_tp_fp_case(labels, predicts)
            for i in range(num_class):
                all_gts[i] += gts[i]
                all_tps[i] += tps[i]
                all_fps[i] += fps[i]
                all_scores[i] += scores[i]
            # summary show case
            if bad_cnt > worst_cnt:
                worst_cnt = bad_cnt
                worst_img = self.plot(v_shapes[bad_idx], v_images[bad_idx], predicts[bad_idx],
                                      labels[bad_idx], threshold=self.obj_thresh)
        all_ap = self.evaluate(all_gts, all_tps, all_fps, all_scores)
        m_ap = 0
        n = 0
        for ap in all_ap:
            if ap > -1:
                m_ap += ap
                n += 1
        if n > 0:
            m_ap /= float(n)

        avg_loss = None
        if total_step > 0:
            avg_loss = total_loss / total_step
        return avg_loss, m_ap, worst_img

    def get_boxes(self, shapes, inputs, predicts, coco=False, nms='nms', cls='sigmoid'):
        batch_size = shapes.shape[0]
        batch_boxes = [None] * batch_size
        _, net_h, net_w, _ = inputs.shape
        for i in range(batch_size):
            image_h, image_w = shapes[i][:2]
            boxes = []
            for j, feature in enumerate([predicts[0][i], predicts[1][i], predicts[2][i]]):
                anchors = self.config['model']['anchors'][(2-j) * 6: (3-j) * 6]
                boxes += decode_netout_mat(feature, anchors, self.obj_thresh, net_h, net_w, cls=cls)
            correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w, coco)
            if nms == 'nms':
                do_nms(boxes, self.nms_thresh)
            elif nms == 'mat_nms':
                do_mat_nms(boxes, self.nms_thresh)
            elif nms == 'fast_nms':
                do_fast_nms(boxes, self.nms_thresh)
            elif nms == 'cv_nms':
                do_cv_nms(boxes, self.nms_thresh)
            else:
                raise Exception('invalid nms function type:{}'.format(nms))
            boxes = [[b.xmin, b.ymin, b.xmax, b.ymax, b.get_score(), b.get_label()]
                     for b in boxes if b.get_score() > 0]
            boxes.sort(key=lambda t: t[4], reverse=True)
            # separate every class
            class_boxes = []
            for _ in range(len(self.config['model']['labels'])):
                class_boxes.append([])
            for box in boxes:
                idx = int(box[5])
                class_boxes[idx].append(box)
            batch_boxes[i] = class_boxes
        return batch_boxes

    def get_labels(self, annotations):
        labels = [None] * annotations.shape[0]
        for i in range(annotations.shape[0]):
            ann = json.loads(annotations[i])
            boxes = []
            for obj in ann['object']:
                idx = self.config['model']['labels'].index(obj['name'])
                box = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], -1.0, idx]
                boxes.append(list(map(float, box)))
            # separate every class
            class_boxes = []
            for _ in range(len(self.config['model']['labels'])):
                class_boxes.append([])
            for box in boxes:
                idx = int(box[5])
                class_boxes[idx].append(box)
            labels[i] = class_boxes
        return labels

    def get_tp_fp_case(self, labels, predicts, threshold=0.5):
        """
        :param labels: batch_size * number_class * [[xmin, ymin, xmax, ymax, used, class_idx]...]]
        :param predicts: batch_size * number_class * [[xmin, ymin, xmax, ymax, used, class_idx]...]]
        :param threshold: iou threshold
        """
        gts = []
        tps = []
        fps = []
        scores = []

        num_class = len(self.config['model']['labels'])
        for i in range(num_class):
            gts.append(0)
            tps.append([])
            fps.append([])
            scores.append([])

        # for summary show case
        bad_case_idx = 0
        bad_case_cnt = 0
        for bad_idx, (label, predict) in enumerate(zip(labels, predicts)):
            for k in range(num_class):
                gts[k] += len(label[k])

            bad_cnt = 0
            for i in range(num_class):
                class_predict = predict[i]
                class_label = label[i]
                for det in class_predict:
                    scores[i].append(det[4])
                    if len(class_label) == 0:
                        fps[i].append(1)
                        tps[i].append(0)
                        if det[4] > 0:
                            bad_cnt += 1
                        continue
                    iou = Evaluator.compute_overlap(np.expand_dims(det[:4], axis=0), np.array(class_label))
                    idx = np.argmax(iou, axis=1)[0]
                    max_iou = iou[0, idx]

                    if max_iou >= threshold and class_label[idx][4] < 0:
                        fps[i].append(0)
                        tps[i].append(1)
                        class_label[idx][4] = 1.0
                    else:
                        fps[i].append(1)
                        tps[i].append(0)
                        if det[4] > 0:  # no suppressed by nms
                            bad_cnt += 1
            if bad_cnt > bad_case_cnt:
                bad_case_cnt = bad_cnt
                bad_case_idx = bad_idx
        return tps, fps, scores, gts, bad_case_cnt, bad_case_idx

    @staticmethod
    def compute_overlap(a, b):
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

        iw = np.minimum(np.expand_dims(a[:, 2], 1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
        ih = np.minimum(np.expand_dims(a[:, 3], 1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

        iw = np.maximum(iw, 0)
        ih = np.maximum(ih, 0)

        ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
        ua = np.maximum(ua, np.finfo(float).eps)
        intersection = iw * ih

        return intersection / ua

    def evaluate(self, all_gts, all_tps, all_fps, all_scores):
        all_ap = []
        for i in range(len(self.config['model']['labels'])):
            gts = all_gts[i]
            tps = all_tps[i]
            fps = all_fps[i]
            scores = all_scores[i]

            if gts == 0:
                if len(tps) == 0:
                    all_ap.append(-1)
                else:
                    all_ap.append(0)
                continue

            tps = np.array(tps)
            fps = np.array(fps)
            scores = np.array(scores)

            indices = np.argsort(-scores)
            fps = fps[indices]
            tps = tps[indices]

            fps = np.cumsum(fps)
            tps = np.cumsum(tps)

            # compute recall and precision
            recall = tps / float(gts)
            precision = tps / np.maximum(tps + fps, np.finfo(np.float64).eps)
            ap = Evaluator.compute_ap(recall, precision)
            all_ap.append(ap)
        return all_ap

    @staticmethod
    def compute_ap(recall, precision):
        """ Compute the average precision, given the recall and precision curves.
        Code originally from https://github.com/rbgirshick/py-faster-rcnn.

        # Arguments
            recall:    The recall curve (list).
            precision: The precision curve (list).
        # Returns
            The average precision as computed in py-faster-rcnn.
        """
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def plot(self, shape, image, predict, label, threshold=0.5):
        h, w = list(map(int, shape[:2]))
        image = cv2.resize(image, (w, h))
        image_pil = Image.fromarray(image[:, :, ::-1])
        # predict box
        for class_predict in predict:
            for box in class_predict:
                if box[4] < threshold:
                    continue
                x_min, y_min, x_max, y_max = list(map(int, box[:4]))
                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                x_max = min(x_max, w)
                y_max = min(y_max, h)
                if x_min >= x_max or y_min >= y_max:
                    continue
                class_name = self.config['model']['labels'][int(box[5])]
                draw_bounding_box_on_image(image_pil, y_min, x_min, y_max, x_max,
                                           color='red',
                                           thickness=1, display_str_list=['{}:{:.2f}'.format(class_name, box[4])],
                                           use_normalized_coordinates=False)
                # cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
                # cv2.putText(image, '{}:{:.2f}'.format(class_name, box[4]), (x_min, y_min),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        # ground truth box
        for class_label in label:
            for box in class_label:
                x_min, y_min, x_max, y_max = list(map(int, box[:4]))
                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                x_max = min(x_max, w)
                y_max = min(y_max, h)
                if x_min >= x_max or y_min >= y_max:
                    continue
                class_name = self.config['model']['labels'][int(box[5])]
                draw_bounding_box_on_image(image_pil, y_min, x_min, y_max, x_max,
                                           color='green',
                                           thickness=1, display_str_list=[class_name],
                                           use_normalized_coordinates=False)
                # cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
                # cv2.putText(image, '{}'.format(class_name), (x_min, y_min),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        np.copyto(image, np.array(image_pil)[:, :, ::-1])
        return image
