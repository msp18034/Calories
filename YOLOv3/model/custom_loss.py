import tensorflow as tf
from tensorflow.python.keras.layers import Layer


class CustomLossLayer(Layer):

    def __init__(self, anchors, max_grid, batch_size, warm_up_batches, ignore_thresh,
                 grid_scale, obj_scale, noobj_scale, xywh_scale, class_scale,
                 **kwargs):
        # make the model settings persistent
        self.ignore_thresh = ignore_thresh
        self.warm_up_batches = warm_up_batches
        self.anchors = tf.constant(anchors, dtype='float', shape=[1, 1, 1, 3, 2])
        self.grid_scale = grid_scale
        self.obj_scale = obj_scale
        self.noobj_scale = noobj_scale
        self.xywh_scale = xywh_scale
        self.class_scale = class_scale
        self.batch_size = batch_size
        self.iou_scale = 1.0
        # focal loss parameters
        self.eps = 1e-14
        self.gamma = 2
        self.alpha = 0.25
        # smooth l1 parameters
        self.sigma = 3.0

        # make a persistent mesh grid
        max_grid_h, max_grid_w = max_grid
        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(max_grid_w), [max_grid_h]), (1, max_grid_h, max_grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))
        self.cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [batch_size, 1, 1, 3, 1])

        super(CustomLossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CustomLossLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        # input_image: [None, net_h, net_w, 3]
        # y_pred: [None, grid_h, grid_w, anchor, 4+1+nb_class]
        # y_true: [None, grid_h, grid_w, anchor, 4+1+nb_class]
        # true_boxes: [None, 1, 1, 1, max_box, 4]
        input_image, y_pred, y_true, true_boxes = x

        # adjust the shape of the y_predict [batch, grid_h, grid_w, 3, 4+1+nb_class]
        y_pred = tf.reshape(y_pred, tf.concat([tf.shape(y_pred)[:3], tf.constant([3, -1])], axis=0))

        # initialize the masks
        object_mask = y_true[..., 4:5]  # [None, grid_h, grid_w, anchor, 1]

        # compute grid factor and net factor
        grid_h = tf.shape(y_true)[1]
        grid_w = tf.shape(y_true)[2]
        grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1, 1, 1, 1, 2])

        net_h = tf.shape(input_image)[1]
        net_w = tf.shape(input_image)[2]
        net_factor = tf.reshape(tf.cast([net_w, net_h], tf.float32), [1, 1, 1, 1, 2])

        """
        Adjust prediction
        """
        pred_box_xy = (self.cell_grid[:, :grid_h, :grid_w, :, :] + tf.sigmoid(y_pred[..., :2]))
        pred_box_wh = y_pred[..., 2:4]
        pred_box_conf = y_pred[..., 4:5]
        pred_box_class = y_pred[..., 5:]

        """
        Adjust ground truth
        """
        true_box_xy = y_true[..., 0:2]
        true_box_wh = y_true[..., 2:4]
        true_box_conf = y_true[..., 4:5]
        true_box_class = y_true[..., 5:]

        """
        Compare each predicted box to all true boxes
        """
        # true_boxes: [center_x, center_y] : grid scale, [w, h] : net scale without divide by anchor
        # normalize both to [0, 1]
        true_xy = true_boxes[..., 0:2] / grid_factor  # [None, 1, 1, 1, max_box, 2]
        true_wh = true_boxes[..., 2:4] / net_factor  # [None, 1, 1, 1, max_box, 2]

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        # normalize pred_xy & pred_wh to [0, 1]
        pred_xy = tf.expand_dims(pred_box_xy / grid_factor, 4)  # [None, grid_h, grid_w, anchor, 1, 2]
        pred_wh = tf.expand_dims(tf.exp(pred_box_wh) * self.anchors / net_factor, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)

        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]  # [None, grid_h, grid_w, anchor, 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]  # [None, 1, 1, 1, max_box]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]  # [None, 1, 1, 1, max_box]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)  # [None, grid_h, grid_w, anchor, max_box]

        best_ious = tf.reduce_max(iou_scores, axis=4, keep_dims=True)  # [None, grid_h, grid_w, anchor, 1]
        conf_mask_false = tf.to_float(best_ious < self.ignore_thresh) * (1 - object_mask)

        """
        Warm-up training
        """
        global_step = tf.train.get_or_create_global_step()
        true_box_xy, true_box_wh, xywh_mask = tf.cond(tf.less(global_step, self.warm_up_batches),
                lambda: [true_box_xy + (0.5 + self.cell_grid[:, :grid_h, :grid_w, :, :]) * (1 - object_mask),
                         true_box_wh, tf.ones_like(object_mask)],
                lambda: [true_box_xy, true_box_wh, object_mask])

        """
        Compare each true box to all anchor boxes
        """
        wh_scale = tf.exp(true_box_wh) * self.anchors / net_factor
        # the smaller the box, the bigger the scale
        wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4)

        loss_xy = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_box_xy-self.cell_grid[:, :grid_h, :grid_w, :, :],
                                                          logits=y_pred[..., :2])
        loss_xy = self.xywh_scale * tf.reduce_sum(loss_xy * wh_scale * xywh_mask, list(range(1, 5)))
        loss_xy = tf.reduce_mean(loss_xy)

        # smooth l1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        sigma_squared = self.sigma ** 2
        regression_diff = tf.abs(true_box_wh - pred_box_wh)
        loss_wh = tf.where(
            tf.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * tf.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )
        loss_wh = self.xywh_scale * tf.reduce_sum(loss_wh * wh_scale * xywh_mask, list(range(1, 5)))
        loss_wh = tf.reduce_mean(loss_wh)

        focal_conf = tf.pow(true_box_conf - tf.sigmoid(pred_box_conf), 2)
        loss_conf = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_box_conf, logits=pred_box_conf)
        loss_conf_positive = self.obj_scale * tf.reduce_sum(loss_conf * object_mask * focal_conf, list(range(1, 5)))
        loss_conf_positive = tf.reduce_mean(loss_conf_positive)
        loss_conf_negative = self.noobj_scale * tf.reduce_sum(loss_conf * conf_mask_false * focal_conf, list(range(1, 5)))
        loss_conf_negative = tf.reduce_mean(loss_conf_negative)
        loss_conf = loss_conf_positive + loss_conf_negative

        # focal_class = tf.pow(true_box_class - tf.nn.softmax(pred_box_class), 2)
        loss_class = tf.losses.softmax_cross_entropy(onehot_labels=true_box_class, logits=pred_box_class,
                                                     reduction=tf.losses.Reduction.NONE)
        loss_class = self.class_scale * tf.reduce_sum(tf.expand_dims(loss_class, 4) * object_mask, list(range(1, 5)))
        loss_class = tf.reduce_mean(loss_class)

        loss = (loss_xy + loss_wh + loss_conf + loss_class) * self.grid_scale

        tf.summary.scalar('loss_total', loss)
        tf.summary.scalar('loss_xy', loss_xy)
        tf.summary.scalar('loss_wh', loss_wh)
        tf.summary.scalar('loss_conf', loss_conf)
        tf.summary.scalar('loss_conf_positive', loss_conf_positive)
        tf.summary.scalar('loss_conf_negative', loss_conf_negative)
        tf.summary.scalar('loss_class', loss_class)

        positive_cnt = tf.reduce_sum(object_mask)
        negative_cnt = tf.reduce_sum(conf_mask_false)
        ignore_cnt = tf.cast(self.batch_size * grid_h * grid_w * 3, tf.float32) - positive_cnt - negative_cnt
        tf.summary.scalar('loss_positive_cnt', positive_cnt)
        tf.summary.scalar('loss_negative_cnt', negative_cnt)
        tf.summary.scalar('loss_ignore_cnt', ignore_cnt)

        # loss = tf.Print(loss, [grid_h, loss, loss_xy, loss_wh, loss_conf, loss_class, tf.reduce_sum(object_mask)],
        #                 message=self.name + ':grid_h, loss, xy, wh, cf, cls, cnt = ', summarize=1000)

        return loss

    def compute_output_shape(self, input_shape):
        return [1]
