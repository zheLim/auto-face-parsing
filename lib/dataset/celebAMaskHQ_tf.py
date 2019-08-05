import tensorflow as tf
import os
import cv2
import numpy as np
from imgaug import augmenters as iaa
from functools import partial
image_feature_description = {
    'height': tf.FixedLenFeature([], tf.int64),
    'width': tf.FixedLenFeature([], tf.int64),
    'depth': tf.FixedLenFeature([], tf.int64),
    'mask_raw': tf.FixedLenFeature([], tf.int64),
    'image_raw': tf.FixedLenFeature([], tf.string),
}



def _parse_image_function(example_proto):
    return

def get_tf_dataset(tf_folder, policy):
    tf_record_files = [os.path.join(tf_folder, fname) for fname in os.listdir(tf_folder)]
    dataset = tf.data.TFRecordDataset(filenames=[tf_record_files])
    dataset = dataset.map(parse_example)
    preprocess_fn = get_augmentation(policy)
    def tf_preprocessing(image, mask):
        image, mask = tf.py_function(preprocess_fn, [image, mask], [np.uint8, np.uint8])
        return image, mask
    dataset = dataset.map(tf_preprocessing)
    return dataset

def parse_example(example_proto):
    features = tf.parse_single_example(example_proto, image_feature_description)
    image = tf.image.decode_jpeg(features['image_raw'])
    mask  = tf.image.decode_png(features['mask_raw'])
    return image, mask

def get_augmentation(policy):
    """
    copy from https://github.com/barisozmen/deepaugment
    :param policy:
    :return:
    """
    scale = policy['scale']
    random_scale_fn = partial(random_scale, max_scale=scale)
    is_rot, max_rot = policy['rotation']
    if not is_rot:
        max_rot = None
    else:
        max_rot = (max_rot-0.5)*np.pi  # [-0.5pi, 0.5pi]
    is_rand_crop, center_ratio = policy['crop']
    if not is_rand_crop:
        center_ratio = None
    else:
        center_ratio = center_ratio-0.5

    random_rotate_crop_fn = partial(random_rotate_crop, max_angle=max_rot, center_ratio=center_ratio)

    iaa_policy = policy['iaa']
    augmentation = []
    for aug_type, magnitude in iaa_policy.items():
        if aug_type == "gaussian-blur":
            augmentation.append(iaa.GaussianBlur(sigma=(0, magnitude * 25.0)))
        elif aug_type == "shear":
            augmentation.append(iaa.Affine(shear=(-90 * magnitude, 90 * magnitude)))
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
            augmentation.append(iaa.Dropout((0.01, max(0.011, magnitude)), per_channel=0.5))
        elif aug_type == "coarse-dropout":
            augmentation.append(iaa.CoarseDropout((0.03, 0.15), size_percent=(0.30, np.log10(magnitude * 3)), per_channel=0.2))
        elif aug_type == "gamma-contrast":
            augmentation.append(iaa.GammaContrast((1-magnitude, 1.0+magnitude)))
        elif aug_type == "brighten":
            augmentation.append(iaa.Add((int(-40 * magnitude), int(40 * magnitude)), per_channel=0.5))  # brighten
        elif aug_type == "invert":
            augmentation.append(iaa.Invert(1.0)) # magnitude not used
        elif aug_type == "fog":
            augmentation.append(iaa.Fog()) # magnitude not used
        elif aug_type == "clouds":
            augmentation.append(iaa.Clouds())  # magnitude not used
        elif aug_type == "super-pixels":  # deprecated
            augmentation.append(iaa.Superpixels(p_replace=(0, magnitude), n_segments=(100, 100)))
        elif aug_type == "elastic-transform":  # deprecated
            augmentation.append(iaa.ElasticTransformation(alpha=(0.0, max(0.5, magnitude * 300)), sigma=5.0))
        elif aug_type == "coarse-salt-pepper":
            augmentation.append(iaa.CoarseSaltAndPepper(p=0.2, size_percent=magnitude))
        elif aug_type == "grayscale":
            augmentation.append(iaa.Grayscale(alpha=(0.0, magnitude)))
        else:
            raise ValueError
    iaa_aug = iaa.Sequential(augmentation)

    def preprocessing(image, mask):
        image, mask = random_scale_fn(image, mask)
        image, mask = iaa_aug(image=image, segmentation_maps=mask)
        image, mask = random_rotate_crop_fn(image, mask)
        return image, mask
    return preprocessing

def random_scale(image, mask, output_size, max_scale=None):
    out_h, out_w = output_size

    if max_scale is not None:
        scale = np.random.uniform(low=0, high=max_scale) * 2
        height, width, _ = image.shape
        dst_h = int(out_h * scale)
        dst_w = int(out_w * scale)
    else:
        dst_h = int(out_h)
        dst_w = int(out_w)
    image = cv2.resize(image, (dst_w, dst_h), interpolation=cv2.INTER_LANCZOS4)
    mask = cv2.resize(mask, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)

    return image, mask

def rotate_scale_crop(image, mask, angle, output_size, center_ratio):
    """
    rotate image and mask
    :param image:
    :param mask:
    :param angle: must be radian
    :param scale: scale related to output size
    :param output_size: final output size
    :param center_ratio: in range [-0.5, 0.5]

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
    bound_h = int(height * abs_cos + width * abs_sin)

    rot_mat[0, 2] += bound_w / 2 - center[0]
    rot_mat[1, 2] += bound_h / 2 - center[1]

    rotated_image = cv2.warpAffine(image, rot_mat, (bound_w, bound_h), flags=cv2.INTER_LANCZOS4, borderValue=128)
    rotated_mask = cv2.warpAffine(mask, rot_mat, (bound_w, bound_h), flags=cv2.INTER_NEAREST, borderValue=0)


    crop_center_x = bound_w//2 + int((bound_w - out_w) * center_ratio[1])
    crop_center_y = bound_h//2 + int((bound_h - out_h) * center_ratio[0])
    crop_x = crop_center_x - out_w // 2
    crop_y = crop_center_y - out_h // 2

    crop_image = rotated_image[crop_y:crop_y+out_h, crop_x:crop_x+out_w]
    crop_mask = rotated_mask[crop_y:crop_y+out_h, crop_x:crop_x+out_w]

    return crop_image, crop_mask


def random_rotate_crop(image, mask, output_size, angle_ratio=None, center_ratio=None):
    angle = 0
    if angle_ratio is not None:
        angle = np.random.uniform(0, angle_ratio) * np.pi - np.pi / 2


    if center_ratio is not None:
        center_ratio = np.random.uniform(0, center_ratio) - 0.5
    else:
        center_ratio = 0 # default is center crop
    crop_img, crop_mask = rotate_scale_crop(image, mask, angle, output_size, center_ratio)

    return crop_img, crop_mask


if __name__ == '__main__':
    get_tf_dataset(tf_folder, policy)
