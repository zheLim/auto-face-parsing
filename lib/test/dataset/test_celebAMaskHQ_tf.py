import os
import numpy as np
import tensorflow as tf
import cv2
import shutil
from lib.dataset.celebAMaskHQ_tf import get_tf_dataset, random_rotate_crop, rotate_crop, random_scale, parse_example
from lib.utils.visualization import mask_coloring


def write_image_mask(image, mask, file_path):
    cv2.imwrite(file_path + '_im.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(file_path + '_mask.png', cv2.cvtColor(mask_coloring(mask), cv2.COLOR_RGB2BGR))


def test_random_rotate_crop():
    policy = {'output_size': (512, 512), 'scale': (False, 0.5), 'rotation': (False, 1), 'crop': (False, 0.5, 0.5)}
    output_size = policy['output_size']
    is_scale, scale = policy['scale']
    is_rot, max_rot = policy['rotation']
    is_rand_crop, crop_x_ratio, crop_y_ratio = policy['crop']

    tf_record_files = '/home/linzhe/data/celebAMask-HQ-tfrecord/train/tfrecord/00-of-10'#[os.path.join(tf_folder, fname) for fname in os.listdir(tf_folder)]
    dataset = tf.data.TFRecordDataset(filenames=[tf_record_files])
    dataset = dataset.map(parse_example)
    idx = 0
    if os.path.exists('/home/linzhe/project/auto-face-parsing/temp'):
        shutil.rmtree('/home/linzhe/project/auto-face-parsing/temp')
    os.makedirs('/home/linzhe/project/auto-face-parsing/temp')
    for image, mask in dataset:

        image = image.numpy()
        mask = mask.numpy()
        image = cv2.resize(image, (1024, 1024))
        mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        path_1024 = '/home/linzhe/project/auto-face-parsing/temp/1024'
        if not os.path.exists(path_1024):
            os.makedirs(path_1024)
        image_center_crop, mask_center_crop = rotate_crop(np.copy(image), np.copy(mask), 0, output_size, 0.5, 0.5, (128, 128, 128))
        write_image_mask(image_center_crop, mask_center_crop,
                         path_1024+'/center' + str(idx))

        image_br_crop, mask_br_crop = rotate_crop(np.copy(image), np.copy(mask), 0, output_size, 0, 0, (128, 128, 128))
        write_image_mask(image_br_crop, mask_br_crop, path_1024+'/br'+str(idx))

        image_tl_crop, mask_tl_crop = rotate_crop(np.copy(image), np.copy(mask), 0, output_size, 1, 1, (128, 128, 128))
        write_image_mask(image_tl_crop, mask_tl_crop, path_1024+'/tl'+str(idx))

        image_rand_1024, mask_rand_1024 = \
            random_rotate_crop(np.copy(image), np.copy(mask), output_size=output_size,
                               max_angle=max_rot, crop_x_ratio=crop_x_ratio, crop_y_ratio=crop_y_ratio)
        write_image_mask(image_rand_1024, mask_rand_1024, path_1024+'/random' + str(idx))

        image = cv2.resize(image, (512, 512))
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        path_512 = '/home/linzhe/project/auto-face-parsing/temp/512'

        if not os.path.exists(path_512):
            os.makedirs(path_512)
        image_center_crop, mask_center_crop = rotate_crop(np.copy(image), np.copy(mask), 0, output_size, 0.5, 0.5, (128, 128, 128))
        write_image_mask(image_center_crop, mask_center_crop,
                         path_512+'/center' + str(idx))

        image_br_crop, mask_br_crop = rotate_crop(np.copy(image), np.copy(mask), 0, output_size, 0, 0, (128, 128, 128))
        write_image_mask(image_br_crop, mask_br_crop, path_512+'/br'+str(idx))

        image_tl_crop, mask_tl_crop = rotate_crop(np.copy(image), np.copy(mask), 0, output_size, 1, 1, (128, 128, 128))
        write_image_mask(image_tl_crop, mask_tl_crop, path_512+'/tl'+str(idx))

        image_rand_512, mask_rand_512 = random_rotate_crop(np.copy(image), np.copy(mask), output_size=output_size, max_angle=max_rot,
                                         crop_x_ratio=crop_x_ratio, crop_y_ratio=crop_y_ratio)
        write_image_mask(image_rand_512, mask_rand_512, path_512+'/random' + str(idx))

        path_256 = '/home/linzhe/project/auto-face-parsing/temp/256'
        if not os.path.exists(path_256):
            os.makedirs(path_256)

        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        image_center_crop, mask_center_crop = rotate_crop(np.copy(image), np.copy(mask), 0, output_size, 0.5, 0.5, (128, 128, 128))
        write_image_mask(image_center_crop, mask_center_crop, path_256+'/center'+str(idx))

        image_center_crop, mask_center_crop = rotate_crop(np.copy(image), np.copy(mask), 0, output_size, 0.5, 0.5, (0, 0, 0))
        write_image_mask(image_center_crop, mask_center_crop,
                         path_256+'/pad_0_center' + str(idx))

        image_br_crop, mask_br_crop = rotate_crop(np.copy(image), np.copy(mask), 0, output_size, 0, 0, (128, 128, 128))
        write_image_mask(image_br_crop, mask_br_crop, path_256+'/br'+str(idx))

        image_tl_crop, mask_tl_crop = rotate_crop(np.copy(image), np.copy(mask), 0, output_size, 1, 1, (128, 128, 128))
        write_image_mask(image_tl_crop, mask_tl_crop, path_256+'/tl'+str(idx))

        image_rot90, mask_rot90 = rotate_crop(np.copy(image), np.copy(mask), 90, output_size, 0.5, 0.5, (128, 128, 128))
        write_image_mask(image_rot90, mask_rot90, path_256+'/rot90'+str(idx))

        image, mask = random_rotate_crop(np.copy(image), np.copy(mask), output_size=output_size, max_angle=max_rot, crop_x_ratio=crop_x_ratio, crop_y_ratio=crop_y_ratio)
        write_image_mask(image, mask, path_256+'/random'+str(idx))

        if idx > 100:
            break
        idx += 1
        pass


def test_random_scale():
    policy = {'output_size': (256, 256), 'scale': (False, 0.5), 'rotation': (False, 1), 'crop': (False, 0.5, 0.5)}
    output_size = policy['output_size']
    is_scale, scale = policy['scale']
    is_rot, max_rot = policy['rotation']
    is_rand_crop, crop_x_ratio, crop_y_ratio = policy['crop']

    idx = 0
    if os.path.exists('/home/linzhe/project/auto-face-parsing/temp'):
        shutil.rmtree('/home/linzhe/project/auto-face-parsing/temp')
    os.makedirs('/home/linzhe/project/auto-face-parsing/temp')

    tf_record_files = '/home/linzhe/data/celebAMask-HQ-tfrecord/train/tfrecord/00-of-10'  # [os.path.join(tf_folder, fname) for fname in os.listdir(tf_folder)]
    dataset = tf.data.TFRecordDataset(filenames=[tf_record_files])
    dataset = dataset.map(parse_example)

    for image, mask in dataset:

        image = image.numpy()
        mask = mask.numpy()

        image_resize_only, mask_resize_only = random_scale(image, mask, output_size)
        write_image_mask(image_resize_only, mask_resize_only, '/home/linzhe/project/auto-face-parsing/temp/resize'+str(idx))

        image_scale_half, mask_scale_half = random_scale(image, mask, output_size, min_scale=0, max_scale=0)
        write_image_mask(image_scale_half, mask_scale_half, '/home/linzhe/project/auto-face-parsing/temp/00'+str(idx))

        image_scale, mask_scale = random_scale(image, mask, output_size, min_scale=1, max_scale=1)
        write_image_mask(image_scale, mask_scale, '/home/linzhe/project/auto-face-parsing/temp/11'+str(idx))

        image_scale_double, mask_scale_double = random_scale(image, mask, output_size, min_scale=0, max_scale=1)
        write_image_mask(image_scale_double, mask_scale_double, '/home/linzhe/project/auto-face-parsing/temp/01'+str(idx))

        image_scale_double, mask_scale_double = random_scale(image, mask, output_size, min_scale=1, max_scale=0)
        write_image_mask(image_scale_double, mask_scale_double, '/home/linzhe/project/auto-face-parsing/temp/10'+str(idx))

        if idx > 100:
            break
        idx += 1
        pass


def test_all():
    no_all_path = '/home/linzhe/project/auto-face-parsing/temp/noall'
    if os.path.exists(no_all_path):
        shutil.rmtree(no_all_path)
    os.makedirs(no_all_path)

    policy = {'output_size': (256, 256), 'scale': (False, 0., 1), 'rotation': (False, 1), 'crop': (False, 0.5, 0.5),
              'padding_value': (0, 0, 0)}
    tf_folder = '/home/linzhe/data/celebAMask-HQ-tfrecord/valid/tfrecord/'
    dataset = get_tf_dataset(tf_folder, policy)

    idx = 0
    for image, mask in dataset:
        image = image.numpy()
        mask = mask.numpy()
        write_image_mask(image, mask, no_all_path + '/' + str(idx))
        if idx > 10:
            break
        idx += 1
        pass

    no_iaa_path = '/home/linzhe/project/auto-face-parsing/temp/noiaa'
    if os.path.exists(no_iaa_path):
        shutil.rmtree(no_iaa_path)
    os.makedirs(no_iaa_path)

    policy = {'output_size': (256, 256), 'scale': (True, 0., 1), 'rotation': (True, 1), 'crop': (True, 0.5, 0.5), 'padding_value': (0, 0, 0)}
    tf_folder = '/home/linzhe/data/celebAMask-HQ-tfrecord/valid/tfrecord/'
    dataset = get_tf_dataset(tf_folder, policy)

    idx = 0
    for image, mask in dataset:
        image = image.numpy()
        mask = mask.numpy()
        write_image_mask(image, mask, no_iaa_path + '/' + str(idx))
        if idx > 10:
            break
        idx += 1
        pass

    iaa_path = '/home/linzhe/project/auto-face-parsing/temp/iaa'
    if os.path.exists(iaa_path):
        shutil.rmtree(iaa_path)
    os.makedirs(iaa_path)
    policy = {'output_size': (256, 256), 'scale': (False, 0., 1), 'rotation': (False, 1), 'crop': (False, 0.5, 0.5), 'padding_value': (0, 0, 0),
              'gaussian-blur': 0.5, 'shear': 0.5, 'horizontal-flip': 0.5, 'vertical-flip': 0.5, 'sharpen': 0.5,
              'emboss': 0.5, 'additive-gaussian-noise': 0.5, 'dropout': 0.5, 'coarse-dropout': 0.5, 'gamma-contrast': 0.5,
              'brighten': 0.5, 'invert': True, 'fog': True, 'clouds':True, 'super-pixels': 0.5, 'elastic-transform': 0.5,
              'coarse-salt-pepper': (0.5, 0.5)}
    tf_folder = '/home/linzhe/data/celebAMask-HQ-tfrecord/valid/tfrecord/'
    dataset = get_tf_dataset(tf_folder, policy)

    idx = 0
    for image, mask in dataset:
        image = image.numpy()
        mask = mask.numpy()
        write_image_mask(image, mask, iaa_path + '/' + str(idx))
        if idx > 100:
            break
        idx += 1

    for iaa_type in ['gaussian-blur', 'shear', 'horizontal-flip', 'vertical-flip', 'sharpen', 'emboss',
                     'additive-gaussian-noise', 'dropout', 'coarse-dropout', 'gamma-contrast', 'brighten', 'invert',
                     'fog', 'clouds', 'super-pixels', 'elastic-transform', 'coarse-salt-pepper', 'grayscale']:
        path = '/home/linzhe/project/auto-face-parsing/temp/' + iaa_type
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        policy = {'output_size': (256, 256), 'scale': (False, 0., 1), 'rotation': (False, 1), 'crop': (False, 0.5, 0.5),
                  'padding_value': (0, 0, 0),
                  iaa_type: 0.5}
        if iaa_type in ['coarse-salt-pepper']:
            policy[iaa_type] = (0.5, 0.5)
        tf_folder = '/home/linzhe/data/celebAMask-HQ-tfrecord/valid/tfrecord/'
        dataset = get_tf_dataset(tf_folder, policy)

        idx = 0
        for image, mask in dataset:
            image = image.numpy()
            mask = mask.numpy()
            write_image_mask(image, mask, path + '/' + str(idx))
            if idx > 10:
                break
            idx += 1


if __name__ == '__main__':
    #test_random_rotate_crop()
    #test_random_scale()
    test_all()