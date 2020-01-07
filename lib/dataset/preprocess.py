import os
import cv2
import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapOnImage
from functools import partial


def get_augmentation(params):
    """
    copy from https://github.com/barisozmen/deepaugment
    :param params:
    :return:
    """
    padding_value = params['PaddingValue']
    output_size = params['OutputSize']

    scale_params = params['Scale']
    if 'disable' in scale_params:
        random_scale_fn = partial(random_scale, output_size=output_size)
    else:
        min_scale = scale_params['min_scale']
        max_scale = scale_params['max_scale']

        random_scale_fn = partial(random_scale, output_size=output_size, min_scale=min_scale, max_scale=max_scale)

    rot_params = params['Rotation']
    if 'disable' in rot_params:
        max_rot_angle = None
    else:
        max_rot_angle = rot_params['max_angle']

    crop_params = params['Crop']
    if 'disable' in crop_params:
        crop_x_ratio = None
        crop_y_ratio = None
    else:
        crop_x_ratio, crop_y_ratio = crop_params['crop_x'], crop_params['crop_y']

    random_rotate_crop_fn = partial(random_rotate_crop, output_size=output_size, max_angle=max_rot_angle,
                                    crop_x_ratio=crop_x_ratio, crop_y_ratio=crop_y_ratio, padding_value=padding_value)

    augmentation = []
    for aug_type, aug_parameters in params.items():

        if aug_type in ['Scale', 'Rotation', 'Crop', 'OutputSize', 'PaddingValue'] or 'disable' in aug_parameters:
            continue
        elif aug_type == "Shear":
            angle = aug_parameters['angle']
            augmentation.append(iaa.Affine(shear=(-90 * angle, 90 * angle), cval=padding_value[0]))
        elif aug_type == "HorizontalFlip":
            augmentation.append(iaa.Fliplr(aug_parameters['probability']))
        elif aug_type == "VerticalFlip":
            augmentation.append(iaa.Flipud(aug_parameters['probability']))
        elif aug_type == "gaussian-blur":
            augmentation.append(iaa.GaussianBlur(sigma=(0, magnitude * 25.0)))
        elif aug_type == "Brighten":
            magnitude = aug_parameters['magnitude']
            augmentation.append(iaa.Multiply((1-magnitude, 1.0+magnitude)))  # brighten
        elif aug_type == "GammaContrast":
            magnitude = aug_parameters['magnitude']
            augmentation.append(iaa.GammaContrast((1-magnitude, 1.0+magnitude)))
        elif aug_type == "Sharpen":
            max_alpha = aug_parameters['max_alpha']
            max_lightness = aug_parameters['max_lightness']
            augmentation.append(iaa.Sharpen(alpha=(0, max_alpha), lightness=(0.50, max_lightness)))
        elif aug_type == "Emboss":
            max_alpha = aug_parameters['max_alpha']
            max_strength = aug_parameters['max_strength']
            augmentation.append(iaa.Emboss(alpha=(0, max_alpha), strength=(0.0, max_strength)))
        elif aug_type == "MotionBlur":
            min_kernel_size = aug_parameters['min_kernel_size']
            interval = aug_parameters['interval']
            max_angle = aug_parameters['max_angle']
            augmentation.append(
                iaa.Emboss(k=(int(min_kernel_size), int(min_kernel_size+interval)), angle=(0.0, max_angle))
            )

        elif aug_type == "fog" and aug_parameters:
            augmentation.append(iaa.Fog())  # magnitude not used
        elif aug_type == "clouds" and aug_parameters:
            augmentation.append(iaa.Clouds())  # magnitude not used

        elif aug_type == "ElasticTransformation":
            max_alpha = aug_parameters['max_alpha']
            sigma = aug_parameters['sigma']
            augmentation.append(iaa.ElasticTransformation(alpha=(0.0, max_alpha), sigma=sigma))
        elif aug_type == "SaltAndPepper":
            probability = aug_parameters['probability']
            per_channel_prob = aug_parameters['per_channel']
            augmentation.append(iaa.SaltAndPepper(probability, per_channel=per_channel_prob))
        elif aug_type == "CoarseSaltAndPepper":
            probability = aug_parameters['probability']
            size_percent = aug_parameters['size_percent ']
            augmentation.append(iaa.CoarseSaltAndPepper(probability, size_percent=size_percent))

        elif aug_type == "Dropout":
            max_probability = aug_parameters['max_probability']
            per_channel_prob = aug_parameters['per_channel_prob ']
            augmentation.append(iaa.Dropout((0.01, max_probability), per_channel=per_channel_prob))
        elif aug_type == "CoarseDropout":
            max_probability = aug_parameters['max_probability']
            size_percent = aug_parameters['size_percent']
            augmentation.append(iaa.CoarseDropout((0.03, max_probability), size_percent=size_percent))

        elif aug_type == "GrayScale":
            max_alpha = aug_parameters['max_alpha']
            augmentation.append(iaa.Grayscale(alpha=(0., max_alpha)))

        else:
            raise ValueError

    if len(augmentation) > 0:
        iaa_aug = iaa.Sequential(augmentation)
    else:
        iaa_aug = None

    def preprocessing(image, mask):
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

    image_pad = np.pad(image_crop, [[pad_top, pad_bottom], [pad_left, pad_right], [0,0]], mode='constant', constant_values=padding_value)
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
        angle = np.random.uniform(-max_angle, max_angle) * 90

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



if __name__ == '__main__':
    pass