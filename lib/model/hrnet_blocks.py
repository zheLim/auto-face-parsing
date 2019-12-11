import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers


class BasicBlock(layers.Layer):
    expansion = 1

    def __init__(self, in_channels, out_channels, strides=1, bn_momentum=0.01, activation='relu'):
        """
        conv-bn-relu-conv-bn
        :param out_channels: number of output channels
        :param stride: stride of first convolution layer
        :param use_transform: use transform layer to match resolution/channels of residual and input x
        :param bn_momentum: momentum of batch normalization layer
        """
        super(BasicBlock, self).__init__()

        self.conv_bn_0 = keras.Sequential(
            [layers.Conv2D(filters=out_channels, kernel_size=3, strides=strides, padding='same', use_bias=False,
                           activation=activation),
             layers.BatchNormalization(momentum=bn_momentum)])
        self.conv_bn_1 = keras.Sequential(
            [layers.Conv2D(filters=out_channels, kernel_size=3, padding='same', use_bias=False),
             layers.BatchNormalization(momentum=bn_momentum)])

        if in_channels != out_channels:
            self.transform_layer = keras.Sequential(
                [layers.Conv2D(filters=out_channels, kernel_size=1, strides=strides, padding='same',
                               use_bias=False),
                 layers.BatchNormalization(momentum=bn_momentum)])
        else:
            self.transform_layer = None
        self.relu = layers.ReLU()

    def call(self, x):
        residual = x
        out0 = self.conv_bn_0(x)
        out1 = self.conv_bn_1(out0)

        if self.transform_layer is not None:
            residual = self.transform_layer(x)
        out = residual + out1
        out = self.relu(out)
        return out


class Bottleneck(layers.Layer):
    expansion = 4

    def __init__(self, in_channels, out_channels, strides=1, expansion=4, bn_momentum=0.01, activation='relu'):
        """
        conv-bn-relu-conv-bn-relu-conv-bn-transform(if needed)-relu
        :param out_channels:
        :param expansion:
        :param strides:
        :param use_transform:
        :param bn_momentum:
        """
        super(Bottleneck, self).__init__()
        self.conv_bn_0 = keras.Sequential(
            [layers.Conv2D(filters=out_channels, kernel_size=1, strides=strides, padding='same', use_bias=False,
                           activation=activation),
             layers.BatchNormalization(momentum=bn_momentum)])
        self.conv_bn_1 = keras.Sequential(
            [layers.Conv2D(filters=out_channels, kernel_size=3, strides=strides, padding='same', use_bias=False,
                           activation=activation),
             layers.BatchNormalization(momentum=bn_momentum)])
        self.conv_bn_2 = keras.Sequential(
            [layers.Conv2D(filters=out_channels * expansion, kernel_size=1, strides=strides, padding='same',
                           use_bias=False),
             layers.BatchNormalization(momentum=bn_momentum)])

        self.transform_layer = keras.Sequential(
            [layers.Conv2D(filters=out_channels*self.expansion, kernel_size=1, strides=strides, padding='same',
                           use_bias=False),
             layers.BatchNormalization(momentum=bn_momentum)])
        self.relu = layers.ReLU()

    def call(self, x):
        residual = x
        out0 = self.conv_bn_0(x)
        out1 = self.conv_bn_1(out0)
        out2 = self.conv_bn_2(out1)

        residual = self.transform_layer(residual)
        out = self.relu(out2 + residual)
        return out


class Branch(layers.Layer):
    def __init__(self, in_channels, out_channels, block, n_blocks, strides=1, use_transform=False, bn_momentum=0.01):
        super(Branch, self).__init__()

        layer_list = [block(in_channels, out_channels)]

        for i in range(1, n_blocks):

            layer_list.append(block(out_channels, out_channels))
        self.branch = keras.Sequential(layer_list)

    def call(self, x):
        branch_out = self.branch(x)
        return branch_out


class MultiResolutionLayer(layers.Layer):
    def __init__(self, n_channels_list, bn_momentum=0.01, activation='relu'):
        """
        fuse feature from different branch with adding
        :param n_branches:
        :param n_channels:
        :param multi_scale_output:
        """
        super(MultiResolutionLayer, self).__init__()
        self.n_branches = len(n_channels_list)
        self.fuse_layers = [[] for branch_i in range(self.n_branches)]
        for branch_i in range(self.n_branches):
            layer = []
            for branch_j in range(self.n_branches):
                if branch_i < branch_j:
                    # resolution of branch i is greater than branch_j
                    # branch_j will be upsample with nearest resize
                    layer.append(keras.Sequential(
                        [layers.Conv2D(filters=n_channels_list[branch_i], kernel_size=1, strides=1, padding='same',
                                       use_bias=False, activation=activation),
                         layers.BatchNormalization(momentum=bn_momentum)]))
                elif branch_i == branch_j:
                    # branch i is branch_j
                    layer.append(None)
                else:
                    # branch_i > branch_j
                    # resolution of branch i is greater than branch_j
                    # needed to be downsample(stride 2 convolution) branch_i - branch_j times
                    downsample_conv = []
                    for k in range(branch_i - branch_j):
                        if k == branch_i - branch_j - 1:
                            downsample_conv.append(
                                keras.Sequential([
                                    layers.Conv2D(filters=n_channels_list[branch_i], kernel_size=3, strides=2,
                                                  padding='same', use_bias=False, activation=activation),
                                    layers.BatchNormalization(momentum=bn_momentum)]))
                        else:
                            downsample_conv.append(
                                keras.Sequential([
                                    layers.Conv2D(filters=n_channels_list[branch_j], kernel_size=3, strides=2,
                                                  padding='same', use_bias=False, activation=activation),
                                    layers.BatchNormalization(momentum=bn_momentum)]))
                    layer.append(keras.Sequential(downsample_conv))
            self.fuse_layers[branch_i] = layer

    def call(self, features):
        fused_features = []
        for branch_current in range(self.n_branches):  # In order to distinguish i & j, use current instead of i.
            feature_current = features[branch_current]
            height = feature_current.shape[1]
            width = feature_current.shape[2]
            for branch_j in range(self.n_branches):
                if branch_current == branch_j:
                    continue
                elif branch_current < branch_j:
                    # resolution of branch i is greater than branch_j
                    feature_j = tf.image.resize(features[branch_j], [height, width], method=tf.image.ResizeMethod.BILINEAR)
                else:
                    feature_j = features[branch_j]
                feature_current = feature_current + self.fuse_layers[branch_current][branch_j](feature_j)
            fused_features.append(feature_current)
        return fused_features


class HighResolutionBlock(layers.Layer):
    def __init__(self, in_channel_list, out_channel_list, block, n_blocks, use_transform=False):
        """
        E.g. input with 3 resolution
                            |-----------|
        x_0 -> branch_0 ... |Multi      | -> x_0
        x_1 -> branch_1 ... |Resolution | -> x_1
        x_2 -> branch_2 ... |Layer      | -> x_2
                            |-----------|
        :param out_channel_list: number of output channels. If len(out_channel_list) > input feature branches,
        new branch will be generated
        """
        super(HighResolutionBlock, self).__init__()
        self.out_chs_list = out_channel_list
        self.branches = []

        for in_chs, out_chs in zip(in_channel_list, out_channel_list):
            branch = Branch(in_chs, out_chs, block, n_blocks, strides=1, use_transform=use_transform, bn_momentum=0.01)
            self.branches.append(branch)
        self.mul_resolution_layer = MultiResolutionLayer(out_channel_list)

    def call(self, features):
        for branch_idx in range(len(features)):
            features[branch_idx] = self.branches[branch_idx](features[branch_idx])
        features = self.mul_resolution_layer(features)
        return features


class DownSampleLayer(layers.Layer):
    """
    Downsample the last layer
    """
    def __init__(self, in_channel_list, out_channel_list, bn_momentum=0.01, activation='relu'):
        super(DownSampleLayer, self).__init__()
        self.pre_branches = len(in_channel_list)
        cur_branches = len(out_channel_list)
        self.downsample_convs = []

        # TODO: make some squeeze & expand layer
        for i in range(cur_branches):
            if i < self.pre_branches:
                if in_channel_list[i] != out_channel_list[i]:
                    self.downsample_convs.append(
                        keras.Sequential([
                            layers.Conv2D(filters=out_channel_list[i], kernel_size=1, strides=1,
                                          padding='same', use_bias=False, activation=activation),
                            layers.BatchNormalization(momentum=bn_momentum)]))
                else:
                    self.downsample_convs.append(None)
            else:
                self.downsample_convs.append(
                    keras.Sequential([
                        layers.Conv2D(filters=out_channel_list[i], kernel_size=3, strides=2,
                                      padding='same', use_bias=False, activation=activation),
                        layers.BatchNormalization(momentum=bn_momentum)])
                )

    def call(self, features):
        for i, d_conv in enumerate(self.downsample_convs):
            if d_conv is None:
                continue  # feature[i] = feature[i]
            if i < self.pre_branches:
                features[i] = d_conv(features[i])
            else:
                features.append(d_conv(features[-1]))

        return features


class Stage(layers.Layer):
    def __init__(self, in_channel_list, out_channel_list, n_hr_blocks, n_blocks, block, bn_momentum=0.01):
        """
                E.g. input with 3 resolution

        x_0 ->   |-------------------|       |-------------------|      x_0 --> x_0
        x_1 ->   |       Hight       |       |       Hight       |      x_1 --> x_1
        x_2 ->   |     Resolution    |  -->  |     Resolution    | -->  x_2 --> x_2
                 |       Block 0     |       |       Block ...   |           | downsample
                 |-------------------|       |-------------------|           >  x_3
        """
        super(Stage, self).__init__()
        pre_branches = len(in_channel_list)
        cur_branches = len(out_channel_list)
        if cur_branches < pre_branches:
            raise ValueError('Current number of branch is smaller than previous number of branch')

        self.hr_blocks = []
        if in_channel_list != out_channel_list[:pre_branches]:
            use_transform = True
        else:
            use_transform = False
        self.hr_blocks.append(HighResolutionBlock(in_channel_list, out_channel_list[:pre_branches], block, n_blocks[0], use_transform))
        self.hr_blocks.extend([
            HighResolutionBlock(out_channel_list[:pre_branches], out_channel_list[:pre_branches], block, n_blocks[i]) for i in range(1, n_hr_blocks)
        ])
        in_channel_list = [block.expansion * chs for chs in in_channel_list]
        self.downsample_layer = DownSampleLayer(in_channel_list, out_channel_list, bn_momentum)

    def call(self, features):
        hr_outs = features
        for h_block in self.hr_blocks:
            hr_outs = h_block(hr_outs)
        downsample_outs = self.downsample_layer(hr_outs)
        return downsample_outs


