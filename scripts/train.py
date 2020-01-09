import cv2
import os
import yaml
import numpy as np
import torch
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from lib.model.seg_hrnet import HighResolutionNet
from lib.dataset.dataset import HelenDataset
from lib.loss.loss import OhemLoss
from lib.loss.metrics import iou
from lib.utils.visualization import visual_image_and_segmentation


def main(params):
    parser = ArgumentParser()
    parser.add_argument('--config', type=str)
    n_classes = 19
    config_file = 'config/hrnet.yaml'
    with open(config_file) as f:
        config_dict = yaml.load(f)

    save_dir = 'outputs/' + config_file.split('/')[1].split('.')[0]
    for kind in ['image', 'model', 'log']:
        this_dir = os.path.join(save_dir, kind)
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)

    model = HighResolutionNet(config_dict['MODEL'])

    train_policy = {'OutputSize': (256, 256), 'Scale': {'disable': True},
                    'Rotation': {'disable': True}, 'Crop': {'disable': True},
                    'PaddingValue': 0}
    train_dataset = HelenDataset(
        '/home/administrator/dataset/helenstar_release',
        train_policy, train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8,
                                               shuffle=True, pin_memory=True,
                                               drop_last=True, num_workers=8)
    valid_policy = {'OutputSize': (256, 256), 'Scale': {'disable': True},
                    'Rotation': {'disable': True}, 'Crop': {'disable': True},
                    'PaddingValue': 0}
    valid_dataset = HelenDataset(
        '/home/administrator/dataset/helenstar_release',
        valid_policy, train=False
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=8, shuffle=False, pin_memory=True,
        drop_last=False, num_workers=8
    )

    #  loss_scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='sparse_categorical_crossentropy')
    #loss_focal = FocalLoss(n_classes, epsilon=1e-7, gamma=2.0, ohem_thresh=5, min_batch_size=16)
    # ohem_loss = OhemLoss(n_classes, ohem_thresh=0.95, batch_size=16, width=256, min_keep=None, epsilon=1e-7, gamma=2.0)
    loss_cce = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    writer = SummaryWriter(f'{save_dir}/log')
    iteration = 0
    model.train()
    model.cuda()
    for epoch in range(100):
        for image, label in train_loader:
            image = image.cuda()
            label = label.cuda()
            iteration += 1
            predict_logits = model(image)
            loss = loss_cce(predict_logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % 200 == 0:
                print('Training loss at iteration %s: %s' % (iteration, loss.item()))
                vis_img_train = visual_image_and_segmentation(image, label, torch.argmax(predict_logits, dim=1))
                cv2.imwrite(f'{save_dir}/image/train_{iteration}.jpg', vis_img_train)
                writer.add_scalar('train loss', float(loss.item()), iteration)

            if iteration % 1000 == 0:
                model.eval()
                iou_valid = []
                with torch.no_grad():
                    for (x_batch_valid, y_batch_valid) in valid_loader:
                        x_batch_valid = x_batch_valid.cuda()
                        y_batch_valid = y_batch_valid.cuda()
                        predict_logits = model(x_batch_valid)
                        predict_mask = torch.argmax(predict_logits, dim=1)
                        iou_res = iou(y_batch_valid, predict_mask)
                        iou_valid.append(iou_res.item())
                    vis_img_valid = visual_image_and_segmentation(x_batch_valid, y_batch_valid, predict_mask)
                    cv2.imwrite(f'{save_dir}/image/valid_{iteration}.jpg', vis_img_valid)

                    print('Validation acc : %s' % (np.mean(iou_valid),))
                    writer.add_scalar('Validation accuracy', np.mean(iou_valid), iteration)
                    writer.flush()
                    # Reset training metrics at the end of each epoch
                    torch.save(model.state_dict(), f'{save_dir}/model/{iteration}')
                model.train()


if __name__ == '__main__':
    params = None
    main(params)