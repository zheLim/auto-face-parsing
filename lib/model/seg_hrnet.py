import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

from lib.model.hrnet_blocks import Stage, BasicBlock, Bottleneck

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(keras.Model):
    def __init__(self, model_config):
        super(HighResolutionNet, self).__init__()

        bn_momentum = model_config['BN_MOMENTUM']
        self.first_conv = keras.Sequential([
            layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(momentum=bn_momentum),
            layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', use_bias=False, activation='relu'),
            layers.BatchNormalization(momentum=bn_momentum)
        ])

        s0_cfg = model_config['STAGE0']
        self.stage0 = Stage([64], s0_cfg['CHANNEL_LIST'],  s0_cfg['NUM_HR_BLOCKS'], s0_cfg['NUM_BLOCKS'],
                            blocks_dict[s0_cfg['BLOCK']], bn_momentum)

        s1_cfg = model_config['STAGE1']
        self.stage1 = Stage(s0_cfg['CHANNEL_LIST'], s1_cfg['CHANNEL_LIST'], s1_cfg['NUM_HR_BLOCKS'],
                            s1_cfg['NUM_BLOCKS'], blocks_dict[s1_cfg['BLOCK']], bn_momentum)

        s2_cfg = model_config['STAGE2']
        self.stage2 = Stage(s1_cfg['CHANNEL_LIST'], s2_cfg['CHANNEL_LIST'], s2_cfg['NUM_HR_BLOCKS'],
                            s2_cfg['NUM_BLOCKS'], blocks_dict[s2_cfg['BLOCK']], bn_momentum)

        s3_cfg = model_config['STAGE3']
        self.stage3 = Stage(s2_cfg['CHANNEL_LIST'], s3_cfg['CHANNEL_LIST'], s3_cfg['NUM_HR_BLOCKS'],
                            s3_cfg['NUM_BLOCKS'], blocks_dict[s3_cfg['BLOCK']], bn_momentum)

        self.out_layer = keras.Sequential([
            layers.Conv2D(filters=sum(s3_cfg['CHANNEL_LIST']), kernel_size=1, strides=1, padding='same',
                          use_bias=False, activation='relu'),
            layers.BatchNormalization(momentum=bn_momentum),
            layers.Conv2D(filters=model_config['NUM_CLASSES'], kernel_size=1, padding='same')
        ])

    def call(self, images):
        images = (images - 128)/2
        out = self.first_conv(images)
        out = self.stage0([out])
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)

        im_h = images.shape[1]
        im_w = images.shape[2]

        feat0 = tf.image.resize(out[0], (im_h, im_w))
        feat1 = tf.image.resize(out[1], (im_h, im_w))
        feat2 = tf.image.resize(out[2], (im_h, im_w))
        feat3 = tf.image.resize(out[3], (im_h, im_w))

        mask = self.out_layer(tf.concat((feat0, feat1, feat2, feat3), axis=3))
        return mask


if __name__ == '__main__':
    import yaml
    import os
    os.chdir('../..')
    with open('config/hrnet.yaml') as f:
        config_dict = yaml.load(f)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    hr_net = HighResolutionNet(config_dict['MODEL'])
    image = tf.constant(np.ones((1, 256, 256, 3), dtype=np.float32))
    mask = hr_net(image)
    hr_net.save_weights('outputs/model/{iter}')
    pass
