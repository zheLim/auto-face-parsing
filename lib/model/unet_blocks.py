import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers


class UnetConv2(layers.Layer):
    def __init__(self, out_size, downsample=False):
        super(UnetConv2, self).__init__()

        self.conv_bn_0 = keras.Sequential(
            [layers.Conv2D(filters=out_size, kernel_size=3, padding='same', use_bias=False, activation='relu'),
             layers.BatchNormalization()])
        stride = 2 if downsample else 1
        self.conv_bn_1 = keras.Sequential(
            [layers.Conv2D(filters=out_size, kernel_size=3, strides=stride, padding='same', use_bias=False, activation='relu'),
             layers.BatchNormalization()])

    def call(self, inputs):
        out_0 = self.conv_bn_0(inputs)
        out_1 = self.conv_bn_1(out_0)
        return out_1


class UnetUp(layers.Layer):
    def __init__(self, out_size):
        super(UnetUp, self).__init__()
        self.conv = UnetConv2(out_size)
        self.up = layers.Conv2DTranspose(out_size, kernel_size=2, strides=2)

    def call(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        return self.conv(tf.concat([inputs1, outputs2], 3))



