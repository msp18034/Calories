import tensorflow as tf


class SummaryHook(tf.train.SummarySaverHook):

    def __init__(self, save_steps=None, save_secs=None, output_dir=None,
                 summary_writer=None, scaffold=None, summary_op=None):
        super(SummaryHook, self).__init__(save_steps, save_secs, output_dir, summary_writer, scaffold, summary_op)

    def before_run(self, run_context):
        if run_context.original_args is None or run_context.original_args.feed_dict is None:
            return
        if 'phase:0' not in run_context.original_args.feed_dict or not run_context.original_args.feed_dict['phase:0']:
            return
        return super(SummaryHook, self).before_run(run_context)

    def after_run(self, run_context, run_values):
        if run_context.original_args is None or run_context.original_args.feed_dict is None:
            return
        if 'phase:0' not in run_context.original_args.feed_dict or not run_context.original_args.feed_dict['phase:0']:
            return
        return super(SummaryHook, self).after_run(run_context, run_values)
