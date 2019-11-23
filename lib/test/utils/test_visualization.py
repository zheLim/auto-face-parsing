import cv2
import os
from lib.dataset.celebAMaskHQ_tf import get_tf_dataset
from lib.model.unet import Unet

from lib.utils.visualization import visual_image_and_segmentation


def test_visualization():
    n_classes = 19
    model = Unet(n_classes)
    tf_folder = '/home/linzhe/dataset/celebAMask-HQ-tfrecord/train/tfrecord/'
    train_policy = {'OutputSize': (256, 256), 'Scale': {'min_scale': 0.5, 'max_scale': 1},
                    'Rotation': {'max_angle': 0.2}, 'Crop': {'crop_x': 0.5, 'crop_y': 0.5}, 'PaddingValue': 0}
    train_dataset = get_tf_dataset(tf_folder, train_policy, shuffle=True)
    valid_policy = {'OutputSize': (256, 256), 'Scale': {'disable': True},
                    'Rotation': {'disable': True}, 'Crop': {'disable': True}, 'PaddingValue': 0}
    valid_dataset = get_tf_dataset(tf_folder, valid_policy, shuffle=False)

    (x_batch_train, y_batch_train) = next(iter(train_dataset))
    vis_img_train = visual_image_and_segmentation(x_batch_train, y_batch_train, y_batch_train)

    (x_batch_valid, y_batch_valid) = next(iter(valid_dataset))
    vis_img_valid = visual_image_and_segmentation(x_batch_valid, y_batch_valid, y_batch_valid)

    print(os.getcwd())
    cv2.imwrite('../../../outputs/train.jpg', vis_img_train)
    cv2.imwrite('../../../outputs/valid.jpg', vis_img_valid)


if __name__ == '__main__':
    test_visualization()