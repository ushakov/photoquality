import tensorflow as tf
from keras.models import Model
from keras.callbacks import Callback
import keras.backend as K
import numpy as np


class SaveSubmodel(Callback):
    """Save a particular part of model every epoch.
    """

    def __init__(self, filepath, model, save_weights_only=False):
        super(SaveSubmodel, self).__init__()
        self.filepath = filepath
        self.saved_model = model
        self.save_weights_only = save_weights_only
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        filepath = self.filepath.format(epoch=epoch, **logs)
        if logs['val_loss'] < self.best:
            self.best = logs['val_loss']
            if self.save_weights_only:
                self.saved_model.save_weights(filepath, overwrite=True)
            else:
                self.saved_model.save(filepath, overwrite=True)


class TensorboardEnabledModel(Model):
    aux_updates = None

    def add_aux_update(self, update):
        if not self.aux_updates:
            self.aux_updates = []
        self.aux_updates.append(update)

    @property
    def updates(self):
        if not self.aux_updates:
            self.aux_updates = []
        super_updates = super(TensorboardEnabledModel, self).updates
        return super_updates + self.aux_updates


class DetailedTensorBoard(Callback):
    def __init__(self, model, log_dir, add_histograms=False):
        self.summaries = tf.Variable('???', dtype=tf.string,
                                     name='summaries_copy')
        self.model = model
        if add_histograms:
            self.make_summaries()
            self.model.add_aux_update(tf.assign(self.summaries,
                                                tf.summary.merge_all()))

        self.writer = tf.summary.FileWriter(log_dir, K.get_session().graph,
                                            flush_secs=10)
        self.iteration = 0
        self.add_histograms = add_histograms

    def make_summaries(self):
        for layer in self.model.layers:
            if hasattr(layer, 'output'):
                tf.summary.histogram('{}_out'.format(layer.name),
                                     layer.output)
        for weight in self.model.trainable_weights:
            name = weight.name.replace(':', '_')
            tf.summary.histogram(name, weight)

    def on_batch_end(self, batch, logs):
        self.iteration += 1
        if self.add_histograms:
            self.writer.add_summary(
                self.summaries.eval(K.get_session()), self.iteration)
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, self.iteration)

    def on_epoch_end(self, batch, logs):
        for name, value in logs.items():
            if not name.startswith('val_'):
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, self.iteration)


def instrument_session_for_debugging(debugger=False, profiler=False):
    sess = K.get_session()

    if debugger:
        from tensorflow.python import debug as tf_debug

        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    if profiler:
        from tensorflow.python.client import timeline
        sess_run = sess.run

        def new_sess_run(self, *args, **kwargs):
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            ret = sess_run(self, *args,
                           options=run_options, run_metadata=run_metadata,
                           **kwargs)

            # Create the Timeline object, and write it to a json
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open('timeline.json', 'w') as f:
                f.write(ctf)
            return ret

        sess.run = new_sess_run

    K.set_session(sess)
