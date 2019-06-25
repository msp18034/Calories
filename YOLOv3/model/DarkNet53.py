import tensorflow as tf
from tensorflow.python.ops.nn_ops import leaky_relu
from tensorflow.python.layers.convolutional import Conv2D
from tensorflow.python.layers.normalization import batch_normalization
from tensorflow.python.keras.layers import UpSampling2D, ZeroPadding2D, concatenate, add
from custom_loss import CustomLossLayer


def conv_block(input, phase, convs, do_skip=True):
    x = input
    count = 0

    for conv in convs:
        if count == (len(convs) - 2) and do_skip:
            skip_connection = x
        count += 1

        if conv['stride'] > 1:
            x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        x = Conv2D(conv['filter'], conv['kernel'], strides=conv['stride'],
                   padding='valid' if conv['stride'] > 1 else 'same',
                   name='conv_' + str(conv['layer_idx']),
                   use_bias=False if conv['norm'] else True)(x)
        if conv['norm']:
            x = batch_normalization(inputs=x, training=phase, name='norm_' + str(conv['layer_idx']))
        if conv['leaky']:
            x = leaky_relu(x, alpha=0.1)

    return add([skip_connection, x]) if do_skip else x


def DarkNet53(max_box_per_image, warm_up_batches, config):
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

    input_image = tf.keras.Input(shape=(None, None, 3), name='input_image')  # net_h, net_w, 3
    true_boxes = tf.keras.Input(shape=(1, 1, 1, max_box_per_image, 4), name='true_boxes')
    true_yolo_1 = tf.keras.Input(shape=(None, None, len(anchors) // 6, 4 + 1 + nb_class), name='true_yolo_1')
    true_yolo_2 = tf.keras.Input(shape=(None, None, len(anchors) // 6, 4 + 1 + nb_class), name='true_yolo_2')
    true_yolo_3 = tf.keras.Input(shape=(None, None, len(anchors) // 6, 4 + 1 + nb_class), name='true_yolo_3')

    # phase = False
    phase = tf.placeholder_with_default(False, shape=(), name='phase')

    # Layer  0 => 4
    x = conv_block(input_image, phase,
                   [{'filter': 32, 'kernel': 3, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 0},
                    {'filter': 64, 'kernel': 3, 'stride': 2, 'norm': True, 'leaky': True, 'layer_idx': 1},
                    {'filter': 32, 'kernel': 1, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 2},
                    {'filter': 64, 'kernel': 3, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 3}])

    # Layer  5 => 8
    x = conv_block(x, phase, [{'filter': 128, 'kernel': 3, 'stride': 2, 'norm': True, 'leaky': True, 'layer_idx': 5},
                              {'filter': 64, 'kernel': 1, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 6},
                              {'filter': 128, 'kernel': 3, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 7}])

    # Layer  9 => 11
    x = conv_block(x, phase, [{'filter': 64, 'kernel': 1, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 9},
                              {'filter': 128, 'kernel': 3, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 10}])

    # Layer 12 => 15
    x = conv_block(x, phase, [{'filter': 256, 'kernel': 3, 'stride': 2, 'norm': True, 'leaky': True, 'layer_idx': 12},
                              {'filter': 128, 'kernel': 1, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 13},
                              {'filter': 256, 'kernel': 3, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 14}])

    # Layer 16 => 36
    for i in range(7):
        x = conv_block(x, phase, [
            {'filter': 128, 'kernel': 1, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 16 + i * 3},
            {'filter': 256, 'kernel': 3, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 17 + i * 3}])

    skip_36 = x

    # Layer 37 => 40
    x = conv_block(x, phase, [{'filter': 512, 'kernel': 3, 'stride': 2, 'norm': True, 'leaky': True, 'layer_idx': 37},
                              {'filter': 256, 'kernel': 1, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 38},
                              {'filter': 512, 'kernel': 3, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 39}])

    # Layer 41 => 61
    for i in range(7):
        x = conv_block(x, phase, [
            {'filter': 256, 'kernel': 1, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 41 + i * 3},
            {'filter': 512, 'kernel': 3, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 42 + i * 3}])

    skip_61 = x

    # Layer 62 => 65
    x = conv_block(x, phase,
                   [{'filter': 1024, 'kernel': 3, 'stride': 2, 'norm': True, 'leaky': True, 'layer_idx': 62},
                    {'filter': 512, 'kernel': 1, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 63},
                    {'filter': 1024, 'kernel': 3, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 64}])

    # Layer 66 => 74
    for i in range(3):
        x = conv_block(x, phase, [
            {'filter': 512, 'kernel': 1, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 66 + i * 3},
            {'filter': 1024, 'kernel': 3, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 67 + i * 3}])

    # Layer 75 => 79
    x = conv_block(x, phase, [{'filter': 512, 'kernel': 1, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 75},
                              {'filter': 1024, 'kernel': 3, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 76},
                              {'filter': 512, 'kernel': 1, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 77},
                              {'filter': 1024, 'kernel': 3, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 78},
                              {'filter': 512, 'kernel': 1, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 79}],
                   do_skip=False)

    # Layer 80 => 82
    pred_yolo_1 = conv_block(x, phase, [
        {'filter': 1024, 'kernel': 3, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 80},
        {'filter': (3 * (5 + nb_class)), 'kernel': 1, 'stride': 1, 'norm': False, 'leaky': False, 'layer_idx': 81}],
                             do_skip=False)

    loss_yolo_1 = CustomLossLayer(anchors[12:], [1 * num for num in max_grid], batch_size,
                                  warm_up_batches, ignore_thresh, grid_scales[0], obj_scale,
                                  noobj_scale, xywh_scale, class_scale
                                  )([input_image, pred_yolo_1, true_yolo_1, true_boxes])

    # Layer 83 => 86
    x = conv_block(x, phase, [{'filter': 256, 'kernel': 1, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 84}],
                   do_skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_61])

    # Layer 87 => 91
    x = conv_block(x, phase, [{'filter': 256, 'kernel': 1, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 87},
                              {'filter': 512, 'kernel': 3, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 88},
                              {'filter': 256, 'kernel': 1, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 89},
                              {'filter': 512, 'kernel': 3, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 90},
                              {'filter': 256, 'kernel': 1, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 91}],
                   do_skip=False)

    # Layer 92 => 94
    pred_yolo_2 = conv_block(x, phase, [
        {'filter': 512, 'kernel': 3, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 92},
        {'filter': (3 * (5 + nb_class)), 'kernel': 1, 'stride': 1, 'norm': False, 'leaky': False, 'layer_idx': 93}],
                             do_skip=False)

    loss_yolo_2 = CustomLossLayer(anchors[6:12], [2 * num for num in max_grid], batch_size,
                                  warm_up_batches, ignore_thresh, grid_scales[1], obj_scale,
                                  noobj_scale, xywh_scale, class_scale
                                  )([input_image, pred_yolo_2, true_yolo_2, true_boxes])

    # Layer 95 => 98
    x = conv_block(x, phase,[{'filter': 128, 'kernel': 1, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 96}],
                   do_skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_36])

    # Layer 99 => 106
    pred_yolo_3 = conv_block(x, phase, [
        {'filter': 128, 'kernel': 1, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 99},
        {'filter': 256, 'kernel': 3, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 100},
        {'filter': 128, 'kernel': 1, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 101},
        {'filter': 256, 'kernel': 3, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 102},
        {'filter': 128, 'kernel': 1, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 103},
        {'filter': 256, 'kernel': 3, 'stride': 1, 'norm': True, 'leaky': True, 'layer_idx': 104},
        {'filter': (3 * (5 + nb_class)), 'kernel': 1, 'stride': 1, 'norm': False, 'leaky': False, 'layer_idx': 105}],
                             do_skip=False)

    loss_yolo_3 = CustomLossLayer(anchors[:6], [4 * num for num in max_grid], batch_size,
                                  warm_up_batches, ignore_thresh, grid_scales[2], obj_scale,
                                  noobj_scale, xywh_scale, class_scale
                                  )([input_image, pred_yolo_3, true_yolo_3, true_boxes])

    output_nodes = [pred_yolo_1, pred_yolo_2, pred_yolo_3]
    return tf.reduce_sum([loss_yolo_1, loss_yolo_2, loss_yolo_3], name='loss'), output_nodes
