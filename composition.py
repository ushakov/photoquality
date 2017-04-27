import random
from keras.applications import vgg19
from keras.models import Model, Sequential
import keras.preprocessing.image as kimg
from keras import layers, optimizers
import keras.backend as K
import gflags
import glog as log
import numpy as np
from PIL import Image, ImageDraw

import ava

FLAGS = gflags.FLAGS

scales = [.5, .6]


class CompositionIterator(ava.AVAIterator):
    num = 0

    def scale_and_pad_image(self, img):
        w, h = img.size
        scale = min(self.target_size[0]*1.0/w, self.target_size[1]*1.0/h)
        img = img.resize((int(w*scale), int(h*scale)))
        offset_x = (self.target_size[0] - img.width) / 2
        offset_y = (self.target_size[1] - img.height) / 2

        pad = Image.new('RGB', self.target_size)
        pad.paste(img, (offset_x, offset_y))
        return pad

    def preprocess_image(self, img):
        try:
            w, h = img.size
            scale = random.choice(scales)
            if random.choice(['corner', 'square']) == 'square':
                sh, ln = sorted([w, h])
                pos = random.randint(0, 2)
                center_x = pos * ln / 4
                center_y = sh / 2
                size = int(sh * scale)
                center_x += random.gauss(0, ln / 16.0)
                center_y += random.gauss(0, (sh - size) / 8.0)
                top_x = int(center_x - size/2)
                top_y = int(center_y - size/2)

                if w < h:
                    top_x, top_y = top_y, top_x
                size_x = size
                size_y = size
            else:
                size_x = int(w * scale)
                size_y = int(h * scale)

                top_x = random.choice([0, w - size_x])
                top_y = random.choice([0, h - size_y])

                top_x = int(random.gauss(top_x, size_x / 8.0))
                top_y = int(random.gauss(top_y, size_y / 8.0))

            top_x = min(max(0, top_x), w - size_x)
            top_y = min(max(0, top_y), h - size_y)

            crop = img.crop((top_x, top_y, top_x + size_x, top_y + size_y))

            orig = self.scale_and_pad_image(img)
            cropped = self.scale_and_pad_image(crop)

            composed = Image.new('RGB', (self.target_size[0] * 2, self.target_size[1]))
            # log.info('composed size=%s', composed.size)
            # log.info('orig size=%s', orig.size)
            # log.info('cropped size=%s', cropped.size)
            composed.paste(orig, (0, 0))
            composed.paste(cropped, (self.target_size[0], 0))

            # if self.dump_to_dir:
            #     draw = ImageDraw.Draw(img)
            #     draw.line((top_x, top_y, top_x + size_x, top_y), fill=(255, 255, 0))
            #     draw.line((top_x + size_x, top_y, top_x + size_x, top_y + size_y), fill=(255, 255, 0))
            #     draw.line((top_x + size_x, top_y + size_y, top_x, top_y + size_y), fill=(255, 255, 0))
            #     draw.line((top_x, top_y + size_y, top_x, top_y), fill=(255, 255, 0))
            #     self.dump_image(img, self.num, 'p')
            #     self.dump_image(composed, self.num, 'r')
            #     self.num += 1
        except Exception:
            import traceback
            traceback.print_exc()
            raise

        return composed

    def make_input(self, imgs, labels):
        arrs = np.stack([kimg.img_to_array(img) for img in imgs])
        arrs = vgg19.preprocess_input(arrs)

        ainp = arrs[:, :, :self.target_size[0], :]
        binp = arrs[:, :, self.target_size[0]:, :]

        if self.dump_to_dir:
            aimg = kimg.array_to_img(ainp[0])
            bimg = kimg.array_to_img(binp[0])
            self.dump_image(aimg, self.num, 'o')
            self.dump_image(bimg, self.num, 'c')
            self.num += 1

        labs = np.ones((ainp.shape[0], ))

        return [ainp, binp], labs


def spatial_pyramid_pooling(inp, sizes):
    pooled = []
    for size in sizes:
        p = layers.AveragePooling2D(pool_size=(size, size),
                                    strides=(size-2, size-2))(inp)
        log.info('p[%s]=%s', size, p)
        p = layers.Flatten()(p)
        pooled.append(p)
    return layers.concatenate(pooled)


def positive(y_true, y_pred):
    return K.mean(K.greater(y_pred, 0), axis=-1)


def hinge_win(y_true, y_pred):
    return K.mean(K.greater(y_pred, 1), axis=-1)


class AVA(ava.AVABase):
    img_size = (256, 256)

    def make_model(self):
        base = vgg19.VGG19(include_top=False, input_shape=(256, 256, 3))
        for layer in base.layers:
            layer.trainable = False

        pooled = spatial_pyramid_pooling(base.output, [8, 5, 4])

        fc1 = layers.Dense(1000, activation='elu')(pooled)

        out = layers.Dense(1)(fc1)

        vfn = Model(base.inputs, out)

        return vfn

    def make_trained_model(self):
        ainp = layers.Input(shape=(256, 256, 3))
        binp = layers.Input(shape=(256, 256, 3))

        ascore = self.model(ainp)
        bscore = self.model(binp)

        minus_bscore = layers.Lambda(lambda x: -x)(bscore)

        diff = layers.add([ascore, minus_bscore])

        log.info('made training model')

        return Model([ainp, binp], diff)

    def compile_model(self, model):
        opt = optimizers.Adam(lr=1e-5)
        model.compile(optimizer=opt, loss='hinge',
                      metrics=[positive, hinge_win])

    def make_iterator(self, dataset):
        positive_ds = [entry for entry in dataset if entry[1] == 1]
        log.info('iterating on %d positive images', len(positive_ds))
        it = CompositionIterator(positive_ds, FLAGS.image_dir,
                                 target_size=self.img_size,
                                 batch_size=self.batch_size)
        # it.dump_to_dir = FLAGS.rundir
        # it.next()
        # log.info('done')
        return it


if __name__ == '__main__':
    cls = AVA()
    cls.main()
