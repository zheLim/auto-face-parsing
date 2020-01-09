import torch
import numpy as np
import cv2
import yaml
import os
from lib.model.seg_hrnet import HighResolutionNet
from lib.utils.visualization import mask_coloring

if __name__ == '__main__':
    os.chdir('..')
    config_file = 'config/hrnet.yaml'
    with open(config_file) as f:
        config_dict = yaml.load(f)
    model = HighResolutionNet(config_dict['MODEL'])
    model.cuda()
    model.eval()
    model.load_state_dict(torch.load('outputs/hrnet/model/49000'))

    alpha = 0.3
    vcap = cv2.VideoCapture('data/facetest2.mp4')
    vwriter = cv2.VideoWriter('outputs/test_result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), vcap.get(cv2.CAP_PROP_FPS), (512, 512))
    with torch.no_grad():
        while True:
            ret, frame = vcap.read()
            if not ret:
                break
            frame = cv2.cvtColor(cv2.resize(frame, (256, 256)), cv2.COLOR_BGR2RGB)

            model_in = torch.from_numpy(np.expand_dims(frame, 0).transpose((0, 3, 1, 2))).cuda()
            logits = model(model_in)
            mask = torch.argmax(logits, dim=1).cpu().numpy().squeeze()
            mask_color = mask_coloring(mask)
            mask_frame = frame * (1-alpha) + mask_color * alpha
            mask_frame = cv2.cvtColor(mask_frame.astype(np.uint8), cv2.COLOR_BGR2RGB)
            mask_frame = cv2.resize(mask_frame, fx=2, fy=2, dsize=None)
            cv2.imshow('frame', mask_frame)
            vwriter.write(mask_frame)
            cv2.waitKey(1)
