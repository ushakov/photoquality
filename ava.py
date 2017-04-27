import os
import sys
import time
import shutil

import gflags
import glog as log
import keras.preprocessing.image as kimg
import keras.backend as K
from keras.applications import imagenet_utils
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau, Callback
from keras.utils import np_utils
import numpy as np

import multi_gpu
import keras_util
import clr_callback

gflags.DEFINE_string('config', '', 'Use named config (default is to read flags)')

gflags.DEFINE_string('load_model', '', 'Model to load and continue training')

gflags.DEFINE_string('train', 'ava-train.txt', 'File with AVA data for training')
gflags.DEFINE_string('val', 'ava-val.txt', 'File with AVA data for validation')

gflags.DEFINE_string('image_dir', '/fast/ava/images', 'Directory containing AVA images')
gflags.DEFINE_string('rundir',
                     'runs/run_' + time.strftime('%Y%m%d-%H%M%S'),
                     'Directory for model checkpoints and logs')

gflags.DEFINE_float('qgap', 1.0,
                    'Gap between positive and negative examples for training')
gflags.DEFINE_integer('gpus', 1,
                      'number of GPUs to use')
gflags.DEFINE_integer('batch_size', 24,
                      'number of GPUs to use')
gflags.DEFINE_integer('prep_threads', 2,
                      'preprocessing threads to start (per 1 GPU)')
FLAGS = gflags.FLAGS


CONFIGS = {
    'home': {
        'image_dir': '/fast/ava/images',
        'rundir': '/home/ushakov/cvimg/runs/run_' + time.strftime('%Y%m%d-%H%M%S'),
        'gpus': 1,
        'batch_size': 72,
    },
    'aws': {
        'image_dir': '/home/ubuntu/ava/images',
        'rundir': '/home/ubuntu/runs/run_' + time.strftime('%Y%m%d-%H%M%S'),
        'gpus': 8,
        'batch_size': 96,
    },
}


def load_ava(fname):
    log.info('Loading dataset from %s', fname)
    ds = []
    with open(fname) as f:
        for line in f:
            fields = map(int, line.strip().split())
            s = 0.0
            num = 0
            for i in range(2, 12):
                s += (i - 1) * fields[i]
                num += fields[i]
            s /= num
            ds.append((fields[1], s))
    log.info('Loaded %d entries', len(ds))
    return ds


def filter_dataset(ds, gap, median=None):
    bin_ds = []
    pos = 0
    neg = 0
    if median is None:
        median = np.median([e[1] for e in ds])
        log.info('Median score = %f', median)
    for id, score in ds:
        if score <= median - gap/2.0:
            bin_ds.append((id, 0))
            neg += 1
        if score > median + gap/2.0:
            bin_ds.append((id, 1))
            pos += 1
    log.info('Filtered %d samples out of %d (with gap=%f) (%d pos/%d neg)',
             len(bin_ds), len(ds), gap, pos, neg)
    return bin_ds, median


class AVAIterator(kimg.Iterator):
    def __init__(self, dataset, image_directory,
                 target_size=(256, 256),
                 batch_size=32):
        self.image_directory = image_directory
        self.dim_ordering = K.image_dim_ordering()
        self.target_size = target_size
        self.image_shape = self.target_size + (3,)
        self.dataset = dataset
        self.dump_to_dir = None
        self.current_position = 0

        super(AVAIterator, self).__init__(len(dataset), batch_size,
                                          False, None)

    def fname(self, id):
        return os.path.join(self.image_directory, '%d.jpg' % id)

    def dump_image(self, img, index, suffix=""):
        #img = kimg.array_to_img(arr, None, scale=True)
        fname = 'dump_{index}_{suffix}_{hash}.png'.format(
            index=index,
            hash=np.random.randint(1e4),
            suffix=suffix)
        img.save(os.path.join(self.dump_to_dir, fname))

    def next(self):
        try:
            return self.__next()
        except Exception:
            import traceback
            traceback.print_exc()
            raise

    def preprocess_image(self, img):
        img = img.resize(self.target_size)
        return img

    def make_input(self, imgs, labels):
        arrs = np.stack([kimg.img_to_array(img) for img in imgs])
        arrs = imagenet_utils.preprocess_input(arrs)

        labs = np.array(labels)
        labs = np_utils.to_categorical(labs, 2)

        return arrs, labs

    def __next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)

        # build batch of image data

        imgs = []
        labels = []
        for j in index_array:
            fname, label = self.dataset[j]
            try:
                img = kimg.load_img(self.fname(fname))
                img = self.preprocess_image(img)
            except:
                log.info('Cannot load image %s', self.fname(fname))
                continue

            imgs.append(img)
            labels.append(label)

        inputs, labels = self.make_input(imgs, labels)

        return inputs, labels


def read_flags():
    """Must remain idempotent!"""
    FLAGS(sys.argv)

    if FLAGS.config:
        cfg = CONFIGS[FLAGS.config]
        for k, v in cfg.items():
            if not FLAGS[k].present:
                setattr(FLAGS, k, v)


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


class ShowSpeed(Callback):
    def __init__(self):
        self.t = time.time()

    def on_batch_end(self, batch, logs):
        now = time.time()
        batch_size = logs.get('size', 1)
        print 'speed',  (now - self.t) * 1000 / batch_size, 'ms/sample'
        self.t = now


class AVABase(object):
    def main(self):
        read_flags()

        if FLAGS.load_model:
            self.model = self.restore_from_saved(FLAGS.load_model)
        else:
            self.model = self.make_model()

        self.model.summary()

        self.train = load_ava(FLAGS.train)
        self.val = load_ava(FLAGS.val)

        log.info('Got %d samples for training, %d for validation',
                 len(self.train), len(self.val))

        self.train, median = filter_dataset(self.train, FLAGS.qgap)
        self.val, _ = filter_dataset(self.val, 0, median)

        log.info('Setting up model for %d GPUs', FLAGS.gpus)
        # self.trainable_model = keras_util.TensorboardEnabledModel(
        #     input=self.model.input, output=self.model.output)
        self.trained_model = self.make_trained_model()
        if FLAGS.gpus > 1:
            self.trained_model = multi_gpu.make_parallel(self.trained_model,
                                                         FLAGS.gpus)
        self.batch_size = FLAGS.batch_size * FLAGS.gpus

        self.compile_model(self.trained_model)

        self.do_train()

    def make_trained_model(self):
        return self.model

    def restore_from_saved(self, name):
        return load_model(name)

    def make_model(self):
        return None

    def compile_model(self, model):
        return None

    def make_iterator(self, dataset):
        return AVAIterator(dataset, FLAGS.image_dir,
                           batch_size=self.batch_size)

    def do_train(self):
        try:
            os.makedirs(FLAGS.rundir)
            import __main__ as main
            shutil.copy(main.__file__, FLAGS.rundir)
        except:
            pass

        train_iterator = self.make_iterator(self.train)
        val_iterator = self.make_iterator(self.val)

        callbacks = []
        callbacks.append(ReduceLROnPlateau(factor=0.5,
                                           patience=2))

        callbacks.append(SaveSubmodel(
            FLAGS.rundir + '/model-{epoch:02d}-{val_loss:.4f}.h5',
            self.model))

        callbacks.append(keras_util.DetailedTensorBoard(self.trained_model,
                                                        FLAGS.rundir))
        callbacks.append(ShowSpeed())

        callbacks.append(clr_callback.CyclicLR())

        total_batches = int(len(self.train) * 1.0 / self.batch_size)
        epoch_size = min(total_batches, int(50000.0 / self.batch_size))
        nb_epochs = int(50.0 * total_batches / epoch_size)

        val_batches = int(len(self.val) * 1.0 / self.batch_size)

        self.trained_model.summary()

        self.trained_model.fit_generator(
            train_iterator, epoch_size, nb_epochs,
            callbacks=callbacks,
            validation_data=val_iterator,
            nb_val_samples=val_batches,
            nb_worker=FLAGS.prep_threads*FLAGS.gpus)
