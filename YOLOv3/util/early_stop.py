import tensorflow as tf
from tensorflow.python.training import session_run_hook
from tensorflow.python.training.session_run_hook import SessionRunArgs


class EarlyStopHook(session_run_hook.SessionRunHook):

    def __init__(self):
        self.early_stop_tensor = EarlyStopHook.get_early_stop_flag()

    def before_run(self, run_context):
        return SessionRunArgs(self.early_stop_tensor)

    def after_run(self, run_context, run_values):
        flag = run_values.results
        if flag:
            run_context.request_stop()

    @staticmethod
    def get_early_stop_flag():
        with tf.variable_scope('early_stop', reuse=tf.AUTO_REUSE):
            flag = tf.get_variable(name='flag', dtype=tf.bool, trainable=False,
                                   initializer=False, collections=[tf.GraphKeys.GLOBAL_VARIABLES])
            return flag

