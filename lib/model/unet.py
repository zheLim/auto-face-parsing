from tensorflow import keras
from lib.model.unet_blocks import UnetConv2, UnetUp


class Unet(keras.Model):
    def __init__(
        self,
        feature_scale=4,
        n_classes=19,
    ):
        super(Unet, self).__init__()
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv2(filters[0], downsample=True)

        self.conv2 = UnetConv2(filters[1], downsample=True)

        self.conv3 = UnetConv2(filters[2], downsample=True)

        self.conv4 = UnetConv2(filters[3], downsample=True)

        self.center = UnetConv2(filters[4], downsample=True)

        # upsampling
        self.up_concat4 = UnetUp(filters[3])
        self.up_concat3 = UnetUp(filters[2])
        self.up_concat2 = UnetUp(filters[1])
        self.up_concat1 = UnetUp(filters[0])

        # final conv (without any concat)
        self.final_up = keras.layers.Conv2DTranspose(filters[0], kernel_size=2, strides=2)

        self.final = keras.layers.Conv2D(n_classes, 1)

    def call(self, inputs):
        inputs = inputs/128 - 1
        conv1 = self.conv1(inputs)  # 128x128
        conv2 = self.conv2(conv1)  # 64x64
        conv3 = self.conv3(conv2)  # 32x32
        conv4 = self.conv4(conv3)  # 16x16

        center = self.center(conv4)

        up4 = self.up_concat4(conv4, center)  # 16x16
        up3 = self.up_concat3(conv3, up4)  # 32x32
        up2 = self.up_concat2(conv2, up3)  # 64x64
        up1 = self.up_concat1(conv1, up2)  # 128x128
        up = self.final_up(up1)
        final = self.final(up)
        return final
