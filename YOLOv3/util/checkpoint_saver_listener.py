import cv2
import datetime
import numpy as np
import tensorflow as tf
from early_stop import EarlyStopHook


class BestCheckpointSaverListener(tf.train.CheckpointSaverListener):

    def __init__(self, saver, config, model_path, validator, summary_writer, early_stop=False, patience=10):
        self.saver = saver
        self.config = config
        self.model_path = model_path
        self.validator = validator
        self.summary_writer = summary_writer
        self.early_stop = early_stop

        self.best_metric = -np.Inf
        self.try_counter = 0
        self.patience = patience

        self.width = 1200
        self.height = 1200

        self.loss_placeholder = tf.placeholder(dtype=tf.float32)
        self.ap_placeholder = tf.placeholder(dtype=tf.float32)
        self.img_placeholder = tf.placeholder(tf.float32, shape=(1, None, None, 3))

        loss = tf.summary.scalar('val_loss', self.loss_placeholder)
        ap = tf.summary.scalar('average_precision', self.ap_placeholder)
        img = tf.summary.image('bad_case', self.img_placeholder, max_outputs=1)

        self.val_merge = tf.summary.merge([loss, ap, img])
        # global early stop flag
        self.early_stop_flag = EarlyStopHook.get_early_stop_flag()
        self.early_stop_update = tf.assign(self.early_stop_flag, True)

    def begin(self):
        print('checkpoint_saver_listener begin')

    def before_save(self, session, global_step_value):
        print('{} BestCheckpointSaverListener:before_save({}) {}'.format('*' * 20, global_step_value, '*' * 20))
        avg_loss, avg_precision, worst_image = self.validator.run(session)
        if worst_image is None:
            print('worst_image from validator is None.')
            return
        h, w = worst_image.shape[:2]
        if (float(self.width) / w) < (float(self.height) / h):
            new_h = (h * self.width) // w
            new_w = self.width
        else:
            new_w = (w * self.height) // h
            new_h = self.height
        worst_image = cv2.resize(worst_image, (new_w, new_h))
        worst_image = worst_image[..., ::-1]
        worst_image = np.expand_dims(worst_image, 0).astype(np.float32)

        if avg_loss is None:
            print('avg_loss from validator is None.')
            return
        val_merge_str = session.run(self.val_merge, feed_dict={self.loss_placeholder: avg_loss,
                                                               self.ap_placeholder: avg_precision,
                                                               self.img_placeholder: worst_image})
        self.summary_writer.add_summary(val_merge_str, global_step_value)
        self.summary_writer.flush()
        print('{}, step={}, avg_loss={}, ap={}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                       global_step_value, avg_loss, avg_precision))

        if avg_precision > self.best_metric:
            self.saver.save(session, self.model_path)
            self.best_metric = avg_precision
            self.try_counter = 0
            print('update best model with iteration {}'.format(global_step_value))
        else:
            self.try_counter += 1
            if self.early_stop and self.try_counter > self.patience:
                _, early_stop_flag = session.run([self.early_stop_update, self.early_stop_flag])
                print('update early_stop_flag = {}'.format(early_stop_flag))

    def should_stop(self, session=None):
        if session is None:
            return False
        early_stop_flag = session.run(self.early_stop_flag)
        return early_stop_flag

    def after_save(self, session, global_step_value):
        # print('{} BestCheckpointSaverListener:after_save({}) {}'.format('*' * 20, global_step_value, '*' * 20))
        pass

    def end(self, session, global_step_value):
        print('{} BestCheckpointSaverListener:end({}) {}'.format('*' * 20, global_step_value, '*' * 20))


