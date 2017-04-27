import glog as log
from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model
import keras.backend as K

import tensorflow as tf


def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        # importing tf here to make sure the model can be loaded later (sic!)
        import tensorflow as tf
        shape = tf.shape(data)
        int_shape = K.int_shape(data)
        part_len = (shape[:1] + parts - 1) / parts
        log.info('part_len=%s', part_len)

        log.info('int_shape[1:] * 0 = %s', ((len(int_shape) - 1) * [0]))

        stride = tf.concat([part_len, (len(int_shape) - 1) * [0]], 0)
        start = stride * idx
        log.info('start=%s stride=%s', start, stride)

        this_part_len = tf.minimum(part_len[0], shape[0] - start[0])
        size = tf.concat([[this_part_len], int_shape[1:]], 0)
        log.info('size=%s', size)
        slc = tf.slice(data, start, size)
        slc = tf.reshape(slc, [-1] + list(int_shape[1:]))

        log.info('slc=%s', slc)

        return slc


    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]

                    slice_n = Lambda(
                        get_slice, output_shape=input_shape,
                        arguments={'idx': i, 'parts': gpu_count})(x)

                    inputs.append(slice_n)
                log.info('scope=%s', scope)
                log.info('inputs=%s', inputs)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # # merge outputs on CPU
    # with tf.device('/cpu:0'):
    merged = []
    for outputs in outputs_all:
        if len(outputs) > 1:
            merged.append(merge(outputs, mode='concat', concat_axis=0))
        else:
            merged.append(outputs[0])

    return Model(input=model.inputs, output=merged)
