import torch
from torch import nn

from lib.model.hrnet_blocks import Stage, BasicBlock, Bottleneck

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):
    def __init__(self, model_config):
        super(HighResolutionNet, self).__init__()

        bn_momentum = model_config['BN_MOMENTUM']
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU()
        )

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

        self.out_layer = nn.Sequential(
            nn.Conv2d(in_channels=sum(s3_cfg['CHANNEL_LIST']), out_channels=sum(s3_cfg['CHANNEL_LIST']), kernel_size=1,
                      stride=1, bias=False),
            nn.BatchNorm2d(sum(s3_cfg['CHANNEL_LIST']), momentum=bn_momentum), nn.ReLU(),
            nn.Conv2d(in_channels=sum(s3_cfg['CHANNEL_LIST']), out_channels=model_config['NUM_CLASSES'], kernel_size=1,
                      padding=0)
        )

    def forward(self, images):
        images = (images - 127.5)/127.5
        out = self.first_conv(images)
        out = self.stage0([out])
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)

        im_h = images.shape[2]
        im_w = images.shape[3]

        feat0 = nn.functional.interpolate(out[0], size=(im_h, im_w))
        feat1 = nn.functional.interpolate(out[1], size=(im_h, im_w))
        feat2 = nn.functional.interpolate(out[2], size=(im_h, im_w))
        feat3 = nn.functional.interpolate(out[3], size=(im_h, im_w))

        mask = self.out_layer(torch.cat((feat0, feat1, feat2, feat3), axis=1))
        return mask


if __name__ == '__main__':
    import yaml
    import os
    import numpy as np
    os.chdir('../..')
    with open('config/hrnet.yaml') as f:
        config_dict = yaml.load(f)

    hr_net = HighResolutionNet(config_dict['MODEL'])
    image = torch.tensor(np.ones((1, 3, 256, 256), dtype=np.float32))
    mask = hr_net(image)
    pass
