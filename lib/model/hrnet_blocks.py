import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, bn_momentum=0.01):
        """
        conv-bn-relu-conv-bn
        :param out_channels: number of output channels
        :param stride: stride of first convolution layer
        :param use_transform: use transform layer to match resolution/channels of residual and input x
        :param bn_momentum: momentum of batch normalization layer
        """
        super(BasicBlock, self).__init__()

        self.conv_bn_0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU()
        )

        self.conv_bn_1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        )

        if in_channels != out_channels:
            self.transform_layer = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0,
                          bias=False),
                nn.BatchNorm2d(out_channels, momentum=bn_momentum))
        else:
            self.transform_layer = None
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out0 = self.conv_bn_0(x)
        out1 = self.conv_bn_1(out0)

        if self.transform_layer is not None:
            residual = self.transform_layer(x)
        out = residual + out1
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, bn_momentum=0.01):
        """
        conv-bn-relu-conv-bn-relu-conv-bn-transform(if needed)-relu
        :param out_channels:
        :param expansion:
        :param strides:
        :param use_transform:
        :param bn_momentum:
        """
        super(Bottleneck, self).__init__()

        self.conv_bn_0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0,
                      bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
        )

        self.conv_bn_1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        )
        self.conv_bn_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion, kernel_size=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(out_channels * self.expansion, momentum=bn_momentum)
        )

        if in_channels != out_channels*self.expansion:
            self.transform_layer = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels * self.expansion, kernel_size=1, stride=stride, padding=0,
                          bias=False),
                nn.BatchNorm2d(out_channels * self.expansion, momentum=bn_momentum))
        else:
            self.transform_layer = None
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out0 = self.conv_bn_0(x)
        out1 = self.conv_bn_1(out0)
        out2 = self.conv_bn_2(out1)
        if self.transform_layer is not None:
            residual = self.transform_layer(residual)
        out = self.relu(out2 + residual)
        return out


class Branch(nn.Module):
    def __init__(self, in_channels, out_channels, block, n_blocks):
        super(Branch, self).__init__()

        layer_list = [block(in_channels, out_channels)]
        for i in range(1, n_blocks):
            layer_list.append(block(out_channels*block.expansion, out_channels))
        self.branch = nn.Sequential(*layer_list)

    def forward(self, x):
        branch_out = self.branch(x)
        return branch_out


class MultiResolutionLayer(nn.Module):
    def __init__(self, n_channels_list, bn_momentum=0.01, activation='relu'):
        """
        fuse feature from different branch with adding
        :param n_branches:
        :param n_channels:
        :param multi_scale_output:
        """
        super(MultiResolutionLayer, self).__init__()
        self.n_branches = len(n_channels_list)
        self.fuse_layers = nn.ModuleList()
        for branch_i in range(self.n_branches):
            layer = nn.ModuleList()
            for branch_j in range(self.n_branches):
                if branch_i < branch_j:
                    # resolution of branch i is greater than branch_j
                    # branch_j will be upsample with nearest resize
                    layer.append(nn.Sequential(
                        nn.Conv2d(in_channels=n_channels_list[branch_j], out_channels=n_channels_list[branch_i],
                                  kernel_size=1, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(n_channels_list[branch_i], momentum=bn_momentum))
                    )

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
                                nn.Sequential(
                                    nn.Conv2d(
                                        in_channels=n_channels_list[branch_j],
                                        out_channels=n_channels_list[branch_i],
                                        kernel_size=3, stride=2, padding=1,
                                        bias=False),
                                    nn.BatchNorm2d(n_channels_list[branch_i], momentum=bn_momentum)))
                        else:
                            downsample_conv.append(
                                nn.Sequential(
                                    nn.Conv2d(in_channels=n_channels_list[branch_j],
                                              out_channels=n_channels_list[branch_j],
                                              kernel_size=3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(n_channels_list[branch_j], momentum=bn_momentum)))

                    layer.append(nn.Sequential(*downsample_conv))
            self.fuse_layers.append(layer)
        pass

    def forward(self, features):
        fused_features = []
        for branch_current in range(self.n_branches):  # In order to distinguish i & j, use current instead of i.
            feature_current = features[branch_current]
            height = feature_current.shape[2]
            width = feature_current.shape[3]
            for branch_j in range(self.n_branches):
                if branch_current == branch_j:
                    continue
                elif branch_current < branch_j:
                    # resolution of branch i is greater than branch_j
                    feature_j = nn.functional.interpolate(features[branch_j], [height, width], mode='bilinear')
                else:
                    feature_j = features[branch_j]
                feat = self.fuse_layers[branch_current][branch_j](feature_j)
                feature_current = feature_current + feat
            fused_features.append(feature_current)
        return fused_features


class HighResolutionBlock(nn.Module):
    def __init__(self, in_channel_list, out_channel_list, block, n_blocks,
                 use_transform=False):
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
        self.branches = nn.ModuleList()

        for in_chs, out_chs in zip(in_channel_list, out_channel_list):
            branch = Branch(in_chs, out_chs, block, n_blocks)
            self.branches.append(branch)
        if len(out_channel_list) > 1:
            self.mul_resolution_layer = MultiResolutionLayer(out_channel_list)
        else:
            self.mul_resolution_layer = None
        pass

    def forward(self, features):
        for branch_idx in range(len(features)):
            features[branch_idx] = self.branches[branch_idx](features[branch_idx])
        if self.mul_resolution_layer is not None:
            features = self.mul_resolution_layer(features)
        return features


class DownSampleLayer(nn.Module):
    """
    Downsample the last layer
    """
    def __init__(self, in_channel_list, out_channel_list, bn_momentum=0.01, activation='relu'):
        super(DownSampleLayer, self).__init__()
        self.pre_branches = len(in_channel_list)
        cur_branches = len(out_channel_list)
        self.downsample_convs = nn.ModuleList()

        # TODO: make some squeeze & expand layer
        for i in range(cur_branches):
            if i < self.pre_branches:
                if in_channel_list[i] != out_channel_list[i]:
                    self.downsample_convs.append(
                        nn.Sequential(
                            nn.Conv2d(in_channels=in_channel_list[i], out_channels=out_channel_list[i], kernel_size=1,
                                      stride=1, padding=0, bias=False),
                            nn.BatchNorm2d(out_channel_list[i], momentum=bn_momentum),
                            nn.ReLU()
                        ))
                else:
                    self.downsample_convs.append(None)
            else:
                self.downsample_convs.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=in_channel_list[-1], out_channels=out_channel_list[i], kernel_size=3,
                                  stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(out_channel_list[i], momentum=bn_momentum))
                )

    def forward(self, features):
        for i, d_conv in enumerate(self.downsample_convs):
            if d_conv is None:
                continue  # feature[i] = feature[i]
            if i < self.pre_branches:
                features[i] = d_conv(features[i])
            else:
                features.append(d_conv(features[-1]))

        return features


class Stage(nn.Module):
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

        self.hr_blocks = nn.ModuleList()
        if in_channel_list != out_channel_list[:pre_branches]:
            use_transform = True
        else:
            use_transform = False
        for i in range(n_hr_blocks):
            self.hr_blocks.append(HighResolutionBlock(in_channel_list, out_channel_list[:pre_branches], block, n_blocks[0], use_transform))
            in_channel_list = [chs * block.expansion for chs in out_channel_list[:pre_branches]]
        self.downsample_layer = DownSampleLayer(in_channel_list, out_channel_list, bn_momentum)

    def forward(self, features):
        hr_outs = features
        for h_block in self.hr_blocks:
            hr_outs = h_block(hr_outs)
        downsample_outs = self.downsample_layer(hr_outs)
        return downsample_outs


