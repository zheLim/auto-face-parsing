import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers


class Block(layers.Layer):
    def __init__(self, out_channels, stride, use_transform=False, bn_momentum=0.01):
        """
        conv-bn-relu-conv-bn
        :param out_channels: number of output channels
        :param stride: stride of first convolution layer
        :param use_transform: use transform layer to match resolution/channels of residual and input x
        :param bn_momentum: momentum of batch normalization layer
        """
        super(Block, self).__init__()

        self.conv_bn_0 = keras.Sequential(
            [layers.Conv2D(filters=out_channels, kernel_size=3, stride=stride, padding='same', use_bias=False),
             layers.BatchNormalization(momentum=bn_momentum)])
        self.conv_bn_1 = keras.Sequential(
            [layers.Conv2D(filters=out_channels, kernel_size=3, padding='same', use_bias=False),
             layers.BatchNormalization(momentum=bn_momentum)])

        self.use_transform = use_transform
        if use_transform:
            self.transform_layer = keras.Sequential(
                [layers.Conv2D(filters=out_channels, kernel_size=1, stride=stride, padding='same', use_bias=False),
                 layers.BatchNormalization(momentum=bn_momentum)])
        self.relu = layers.Relu()

    def call(self, x):
        residual = x
        out0 = self.relu(self.conv_bn_0(x))
        out1 = self.conv_bn_1(out0)

        if self.use_transform:
            residual = self.transform_layer(residual)
        out = residual + out1
        out = self.relu(out)
        return out


class Bottleneck(layers.Layer):
    def __init__(self, out_channels, expansion, stride, use_transform=False, bn_momentum=0.01):
        """
        conv-bn-relu-conv-bn-relu-conv-bn-transform(if needed)-relu
        :param out_channels:
        :param expansion:
        :param stride:
        :param use_transform:
        :param bn_momentum:
        """
        super(Bottleneck, self).__init__()
        self.conv_bn_0 = keras.Sequential(
            [layers.Conv2D(filters=out_channels, kernel_size=1, stride=stride, padding='same', use_bias=False),
             layers.BatchNormalization(momentum=bn_momentum)])
        self.conv_bn_1 = keras.Sequential(
            [layers.Conv2D(filters=out_channels, kernel_size=3, stride=stride, padding='same', use_bias=False),
             layers.BatchNormalization(momentum=bn_momentum)])
        final_out_chs = out_channels * expansion
        self.conv_bn_2 = keras.Sequential(
            [layers.Conv2D(filters=final_out_chs, kernel_size=1, stride=stride, padding='same', use_bias=False),
             layers.BatchNormalization(momentum=bn_momentum), layers.Relu()])

        self.use_transform = use_transform
        if use_transform:
            self.transform_layer = keras.Sequential(
                [layers.Conv2D(filters=out_channels, kernel_size=1, stride=stride, padding='same', use_bias=False),
                 layers.BatchNormalization(momentum=bn_momentum)])
        self.relu = layers.Relu()

    def call(self, x):
        residual = x
        out0 = self.relu(self.conv_bn_0(x))
        out1 = self.relu(self.conv_bn_1(out0))
        out2 = self.conv_bn_2(out1)
        if self.use_transform:
            residual = self.transform_layer(residual)
        out = self.relu(out2 + residual)
        return out


class HighResolutionBlock(layers.Layer):
    def __init__(self, n_branches, out_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionBlock, self).__init__()

