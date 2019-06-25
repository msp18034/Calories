import os
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer


class BestSaverHook(tf.train.CheckpointSaverHook):

    def __init__(self, checkpoint_dir, save_secs=None, save_steps=None, saver=None,
                 checkpoint_basename="model.ckpt", scaffold=None, listeners=None):

        self.saver_listener = listeners[0]
        super(BestSaverHook, self).__init__(checkpoint_dir, save_secs, save_steps, saver,
                                            checkpoint_basename, scaffold, listeners)

        logging.info("Create CheckpointSaverHook.")
        if saver is not None and scaffold is not None:
            raise ValueError("You cannot provide both saver and scaffold.")
        self._saver = saver
        self._checkpoint_dir = checkpoint_dir
        self._save_path = os.path.join(checkpoint_dir, checkpoint_basename)
        self._scaffold = scaffold
        self._timer = SecondOrStepTimer(every_secs=save_secs,
                                        every_steps=save_steps)
        self._listeners = listeners or []

        print('__init__ listeners:{}, {}'.format(len(listeners), len(self._listeners)))

    # def after_run(self, run_context, run_values):
    #     print('EarlyStoppingHook:{}'.format(run_values.results))
    #     super(EarlyStoppingHook, self).after_run(run_context, run_values)
    #     if self.saver_listener.should_stop():
    #         run_context.request_stop()

    def after_run(self, run_context, run_values):
        # print('EarlyStoppingHook:{}'.format(run_values.results))
        stale_global_step = run_values.results
        if self._timer.should_trigger_for_step(stale_global_step+1):
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)
                self._save(run_context.session, global_step)

        if self.saver_listener.should_stop(run_context.session):
            print('early stop')
            run_context.request_stop()

    def _save(self, session, step):
        """Saves the latest checkpoint."""
        self.saver_listener.before_save(session, step)
        self._get_saver().save(session, self._save_path, global_step=step)
        self._summary_writer.add_session_log(SessionLog(status=SessionLog.CHECKPOINT, checkpoint_path=self._save_path), step)
        self.saver_listener.after_save(session, step)
        logging.info("Saving checkpoints for %d into %s.", step, self._save_path)


