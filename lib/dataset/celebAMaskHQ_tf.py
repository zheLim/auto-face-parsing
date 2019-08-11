import tensorflow as tf
import os
import cv2
import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapOnImage
from functools import partial

image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'mask_raw': tf.io.FixedLenFeature([], tf.string),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}


def _parse_image_function(example_proto):
    return


def parse_example(example_proto):
    features = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.io.decode_jpeg(features['image_raw'])
    mask = tf.io.decode_png(features['mask_raw'])
    return image, mask


def get_augmentation(policy):
    """
    copy from https://github.com/barisozmen/deepaugment
    :param policy:
    :return:
    """
    padding_value = policy['padding_value']
    output_size = policy['output_size']
    is_scale, min_scale, max_scale = policy['scale']
    if not is_scale:
        random_scale_fn = partial(random_scale, output_size=output_size)
    else:
        random_scale_fn = partial(random_scale, output_size=output_size, min_scale=min_scale, max_scale=max_scale)

    is_rot, max_rot = policy['rotation']
    if not is_rot:
        max_rot = None

    is_rand_crop, crop_x_ratio, crop_y_ratio = policy['crop']
    if not is_rand_crop:
        crop_x_ratio = None
        crop_y_ratio = None

    random_rotate_crop_fn = partial(random_rotate_crop, output_size=output_size, max_angle=max_rot,
                                    crop_x_ratio=crop_x_ratio, crop_y_ratio=crop_y_ratio, padding_value=padding_value)

    augmentation = []
    for aug_type, magnitude in policy.items():
        if aug_type == "gaussian-blur":
            augmentation.append(iaa.GaussianBlur(sigma=(0, magnitude * 25.0)))
        elif aug_type == "shear":
            augmentation.append(iaa.Affine(shear=(-90 * magnitude, 90 * magnitude), cval=padding_value[0]))
        elif aug_type == "horizontal-flip":
            augmentation.append(iaa.Fliplr(magnitude))
        elif aug_type == "vertical-flip":
            augmentation.append(iaa.Flipud(magnitude))
        elif aug_type == "sharpen":
            augmentation.append(iaa.Sharpen(alpha=(0, 1.0), lightness=(0.50, 5 * magnitude)))
        elif aug_type == "emboss":
            augmentation.append(iaa.Emboss(alpha=(0, 1.0), strength=(0.0, 20.0 * magnitude)))
        elif aug_type == "additive-gaussian-noise":
            augmentation.append(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, magnitude * 255), per_channel=0.5))
        elif aug_type == "dropout":
            augmentation.append(iaa.Dropout((0.01, magnitude*0.1), per_channel=0.5))
        elif aug_type == "coarse-dropout":
            augmentation.append(iaa.CoarseDropout((0.03, 0.15), size_percent=(0.30, np.log10(magnitude * 3)), per_channel=0.2))
        elif aug_type == "gamma-contrast":
            augmentation.append(iaa.GammaContrast((1-magnitude, 1.0+magnitude)))
        elif aug_type == "brighten":
            augmentation.append(iaa.Multiply((1-magnitude, 1.0+magnitude)))  # brighten
        elif aug_type == "invert":
            augmentation.append(iaa.Invert(p=magnitude)) # magnitude not used
        elif aug_type == "fog":
            augmentation.append(iaa.Fog()) # magnitude not used
        elif aug_type == "clouds":
            augmentation.append(iaa.Clouds())  # magnitude not used
        elif aug_type == "super-pixels":  # deprecated
            augmentation.append(iaa.Superpixels(p_replace=(0, magnitude), n_segments=(100, 100)))
        elif aug_type == "elastic-transform":  # deprecated
            augmentation.append(iaa.ElasticTransformation(alpha=(0.0, max(0., magnitude * 5)), sigma=0.25))
        elif aug_type == "coarse-salt-pepper":
            augmentation.append(iaa.CoarseSaltAndPepper(p=magnitude[0], size_percent=magnitude[1]))
        elif aug_type == "grayscale":
            augmentation.append(iaa.Grayscale(alpha=(0., magnitude)))
        elif aug_type in ['scale', 'rotation', 'crop', 'output_size', 'padding_value']:
            pass
        else:
            raise ValueError
    if len(augmentation) > 0:
        iaa_aug = iaa.Sequential(augmentation)
    else:
        iaa_aug = None

    def preprocessing(image, mask):
        image = image.numpy()
        mask = mask.numpy()
        image, mask = random_scale_fn(image, mask)
        if iaa_aug is not None:
            image, segmap = iaa_aug(image=image, segmentation_maps=SegmentationMapOnImage(mask, shape=mask.shape,
                                                                                          nb_classes=19))
            mask = segmap.get_arr_int()
        image, mask = random_rotate_crop_fn(image, mask)
        return image, mask
    return preprocessing


def random_scale(image, mask, output_size, min_scale=None, max_scale=None):
    out_h, out_w = output_size

    if max_scale is not None:
        scale = np.random.uniform(low=max(1-min_scale, 0.1), high=1 + max_scale)
        height, width, _ = image.shape
        dst_h = int(out_h * scale)
        dst_w = int(out_w * scale)
    else:
        dst_h = int(out_h)
        dst_w = int(out_w)
    image = cv2.resize(image, (dst_w, dst_h), interpolation=cv2.INTER_LANCZOS4)
    mask = cv2.resize(mask, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)

    return image, mask


def rotate_crop(image, mask, angle, output_size, crop_x_ratio, crop_y_ratio, padding_value=(128, 128, 128)):
    """
    rotate image and mask
    :param image:
    :param mask:
    :param angle: must be radian
    :param scale: scale related to output size
    :param output_size: final output size
    :param crop_x_ratio: in range [0, 1]
    :param crop_y_ratio: in range [0, 1]

    :return:
    """

    height, width, _ = image.shape
    out_h, out_w = output_size
    center = (image.shape[1]//2, image.shape[0]//2)

    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)

    abs_cos = abs(rot_mat[0, 0])
    abs_sin = abs(rot_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_sin + width * abs_cos)

    rot_mat[0, 2] += bound_w / 2 - center[0]
    rot_mat[1, 2] += bound_h / 2 - center[1]

    rotated_image = cv2.warpAffine(image, rot_mat, (bound_w, bound_h), flags=cv2.INTER_LANCZOS4, borderValue=padding_value)
    rotated_mask = cv2.warpAffine(mask, rot_mat, (bound_w, bound_h), flags=cv2.INTER_NEAREST, borderValue=0)

    crop_center_x = int(bound_w * crop_x_ratio)
    crop_center_y = int(bound_h * crop_y_ratio)

    crop_x = crop_center_x - out_w // 2
    crop_y = crop_center_y - out_h // 2
    pad_top = max(0, -crop_y)
    pad_bottom = max(0, crop_y+out_h - bound_h)
    pad_left = max(0, -crop_x)
    pad_right = max(0, crop_x + out_w - bound_w)

    image_crop = rotated_image[max(0, crop_y):crop_y+out_h, max(0, crop_x):crop_x+out_w]
    mask_crop = rotated_mask[max(0, crop_y):crop_y+out_h, max(0, crop_x):crop_x+out_w]

    image_pad = np.pad(image_crop, [[pad_top, pad_bottom], [pad_left, pad_right], [0,0]], mode='constant', constant_values=padding_value[0])
    mask_pad = np.pad(mask_crop, [[pad_top, pad_bottom], [pad_left, pad_right]], mode='constant', constant_values=0)

    return image_pad, mask_pad


def random_rotate_crop(image, mask, output_size, max_angle=None, crop_x_ratio=None, crop_y_ratio=None, padding_value=(128, 128, 128)):
    """

    :param image:
    :param mask:
    :param output_size:
    :param max_angle:  range [0, 1]
    :param crop_x_ratio: range [0, 1]
    :param crop_y_ratio: range [0, 1]
    :param padding_value:
    :return:
    """
    angle = 0
    if max_angle is not None:
        angle = np.random.uniform(0, max_angle) * 180 - 180 / 2

    if crop_x_ratio is not None:
        crop_x_ratio = crop_x_ratio / 2
        crop_y_ratio = crop_y_ratio / 2

        crop_x_ratio = np.random.uniform(0.5-crop_x_ratio, 0.5+crop_x_ratio)
        crop_y_ratio = np.random.uniform(0.5-crop_y_ratio, 0.5+crop_y_ratio)

    else:
        crop_x_ratio = 0.5 # default is center crop
        crop_y_ratio = 0.5
    crop_img, crop_mask = rotate_crop(image, mask, angle, output_size, crop_x_ratio, crop_y_ratio, padding_value)

    return crop_img, crop_mask


def get_tf_dataset(tf_folder, policy, batch_size=64):
    tf_record_files = [os.path.join(tf_folder, fname) for fname in os.listdir(tf_folder)]
    dataset = tf.data.TFRecordDataset(filenames=[tf_record_files])
    dataset = dataset.map(parse_example)
    preprocess_fn = get_augmentation(policy)

    def tf_preprocessing(image, mask):
        image, mask = tf.py_function(preprocess_fn, [image, mask], [tf.uint8, tf.uint8])
        return image, mask
    dataset = dataset.map(tf_preprocessing, tf.data.experimental.AUTOTUNE)
    return dataset.repeat().batch(batch_size)


if __name__ == '__main__':
    policy = {'scale': (True, 0.5), 'rotation': (True, 1), 'crop': (True, 1), 'gaussian-blur': (True, 1),
              'shear': (True, 1), 'rotation': (True, 1), 'rotation': (True, 1), 'rotation': (True, 1)}
    get_tf_dataset(tf_folder, policy)
