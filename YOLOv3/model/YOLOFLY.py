import tensorflow as tf
from tensorflow.contrib.layers.python.layers.layers import conv2d
from tensorflow.contrib.layers.python.layers.layers import batch_norm
from tensorflow.contrib.layers.python.layers.layers import conv2d_transpose
from custom_loss import CustomLossLayer
from BaseNet import BaseNet


def YOLOFLY(max_box_per_image, warm_up_batches, config):

    labels = list(config['model']['labels'])
    nb_class = len(labels)
    anchors = config['model']['anchors']
    grid_scales = config['train']['grid_scales']
    obj_scale = config['train']['obj_scale'],
    noobj_scale = config['train']['noobj_scale'],
    xywh_scale = config['train']['xywh_scale'],
    class_scale = config['train']['class_scale']
    batch_size = config['train']['batch_size']
    ignore_thresh = config['train']['ignore_thresh']
    max_grid = [config['model']['max_input_size'], config['model']['max_input_size']]

    input_image = tf.keras.Input(shape=(None, None, 3), name='input_image')
    true_boxes = tf.keras.Input(shape=(1, 1, 1, max_box_per_image, 4), name='true_boxes')
    true_yolo_1 = tf.keras.Input(shape=(None, None, len(anchors) // 6, 4 + 1 + nb_class), name='true_yolo_1')
    true_yolo_2 = tf.keras.Input(shape=(None, None, len(anchors) // 6, 4 + 1 + nb_class), name='true_yolo_2')
    true_yolo_3 = tf.keras.Input(shape=(None, None, len(anchors) // 6, 4 + 1 + nb_class), name='true_yolo_3')

    # phase = False
    phase = tf.placeholder_with_default(False, shape=(), name='phase')
    normalizer_params = {'is_training': phase, 'scale': True}

    # tiny dark-net as backbone network
    y, s_skip_15, s_skip_10 = BaseNet(input_image, normalizer_params)

    # Layer 16
    y = conv2d(y, 256, 3, stride=1, padding='SAME', scope='layer_16',
               activation_fn=tf.nn.leaky_relu,
               normalizer_fn=batch_norm,
               normalizer_params=normalizer_params)
    s_skip_16 = y

    # Layer 17
    s_skip_15 = conv2d(s_skip_15, 256, 1, stride=1, padding='SAME', scope='layer_17',
                       activation_fn=tf.nn.leaky_relu,
                       normalizer_fn=batch_norm,
                       normalizer_params=normalizer_params)

    y = conv2d_transpose(y, 256, 3, stride=2, padding='SAME', scope='trans_17',
                         activation_fn=tf.nn.leaky_relu,
                         normalizer_fn=batch_norm,
                         normalizer_params=normalizer_params)

    y = tf.add(y, s_skip_15)

    # Layer 18
    y = conv2d(y, 256, 3, stride=1, padding='SAME', scope='layer_18',
               activation_fn=tf.nn.leaky_relu,
               normalizer_fn=batch_norm,
               normalizer_params=normalizer_params)
    s_skip_18 = y

    # Layer 19
    s_skip_10 = conv2d(s_skip_10, 256, 1, stride=1, padding='SAME', scope='layer_19',
                       activation_fn=tf.nn.leaky_relu,
                       normalizer_fn=batch_norm,
                       normalizer_params=normalizer_params)

    y = conv2d_transpose(y, 256, 3, stride=2, padding='SAME', scope='trans_19',
                         activation_fn=tf.nn.leaky_relu,
                         normalizer_fn=batch_norm,
                         normalizer_params=normalizer_params)

    y = tf.add(y, s_skip_10)

    # Layer 20
    y = conv2d(y, 256, 3, stride=1, padding='SAME', scope='layer_20',
               activation_fn=tf.nn.leaky_relu,
               normalizer_fn=batch_norm,
               normalizer_params=normalizer_params)

    # Layer 21
    s_pred_yolo_3 = conv2d(y, 3 * (4 + 1 + nb_class), 1, stride=1, padding='SAME',
                           scope='layer_21', activation_fn=None)

    # Layer Loss 3
    s_loss_yolo_3 = CustomLossLayer(anchors[:6], [4 * num for num in max_grid], batch_size, warm_up_batches,
                                    ignore_thresh, grid_scales[2], obj_scale, noobj_scale, xywh_scale,
                                    class_scale)([input_image, s_pred_yolo_3, true_yolo_3, true_boxes])

    # Layer 22
    y = conv2d(y, 256, 3, stride=2, padding='SAME', scope='layer_22',
               activation_fn=tf.nn.leaky_relu,
               normalizer_fn=batch_norm,
               normalizer_params=normalizer_params)

    # Layer 23
    s_skip_18 = conv2d(s_skip_18, 256, 1, stride=1, padding='SAME', scope='layer_23',
                       activation_fn=tf.nn.leaky_relu,
                       normalizer_fn=batch_norm,
                       normalizer_params=normalizer_params)

    y = tf.add(y, s_skip_18)

    # Layer 24
    y = conv2d(y, 256, 3, stride=1, padding='SAME', scope='layer_24',
               activation_fn=tf.nn.leaky_relu,
               normalizer_fn=batch_norm,
               normalizer_params=normalizer_params)

    # Layer 25
    s_pred_yolo_2 = conv2d(y, 3 * (4 + 1 + nb_class), 1, stride=1, padding='SAME',
                           scope='layer_25', activation_fn=None)

    # Layer Loss 2
    s_loss_yolo_2 = CustomLossLayer(anchors[6:12], [2 * num for num in max_grid], batch_size, warm_up_batches,
                                    ignore_thresh, grid_scales[1], obj_scale, noobj_scale, xywh_scale,
                                    class_scale)([input_image, s_pred_yolo_2, true_yolo_2, true_boxes])

    # Layer 26
    y = conv2d(y, 256, 3, stride=2, padding='SAME', scope='layer_26',
               activation_fn=tf.nn.leaky_relu,
               normalizer_fn=batch_norm,
               normalizer_params=normalizer_params)

    # Layer 27
    s_skip_16 = conv2d(s_skip_16, 256, 1, stride=1, padding='SAME', scope='layer_27',
                       activation_fn=tf.nn.leaky_relu,
                       normalizer_fn=batch_norm,
                       normalizer_params=normalizer_params)

    y = tf.add(y, s_skip_16)

    # Layer 28
    y = conv2d(y, 256, 3, stride=1, padding='SAME', scope='layer_28',
               activation_fn=tf.nn.leaky_relu,
               normalizer_fn=batch_norm,
               normalizer_params=normalizer_params)

    # Layer 29
    s_pred_yolo_1 = conv2d(y, 3 * (4 + 1 + nb_class), 1, stride=1, padding='SAME',
                           scope='layer_29', activation_fn=None)

    # Layer Loss 1
    s_loss_yolo_1 = CustomLossLayer(anchors[12:], [1 * num for num in max_grid], batch_size, warm_up_batches,
                                    ignore_thresh, grid_scales[0], obj_scale, noobj_scale, xywh_scale,
                                    class_scale)([input_image, s_pred_yolo_1, true_yolo_1, true_boxes])

    s_output_nodes = [s_pred_yolo_1, s_pred_yolo_2, s_pred_yolo_3]
    s_loss = tf.reduce_sum([s_loss_yolo_1, s_loss_yolo_2, s_loss_yolo_3], name="loss")
    return s_loss, s_output_nodes
