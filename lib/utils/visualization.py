import numpy as np
import tensorflow as tf
import cv2

part_colors = [[0, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]


def mask_coloring(mask):
    mask_colored = np.zeros_like(mask)
    if len(mask_colored.shape) == 2:
        mask_colored = mask_colored[:, :, np.newaxis]
    mask_colored = np.tile(mask_colored, [1, 1, 3])
    mask = np.squeeze(mask)
    for label in range(1, 19):
        mask_colored[mask == label] = part_colors[label]
    return mask_colored


def visual_image_and_segmentation(batch_x, batch_mask, batch_pred_mask):
    if not isinstance(batch_x, np.ndarray):
        batch_x = batch_x.numpy()
        batch_mask = batch_mask.numpy()
        batch_pred_mask = batch_pred_mask.numpy()
    if batch_pred_mask.dtype != np.uint8:
        batch_pred_mask = batch_pred_mask.astype(np.uint8)
    masks_colored = []
    for i in range(batch_mask.shape[0]):
         masks_colored.append(mask_coloring(batch_mask[i]))
    masks_colored = np.concatenate(masks_colored)
    masks_colored = cv2.cvtColor(masks_colored, cv2.COLOR_BGR2RGB)

    pred_masks_colored = []
    for i in range(batch_pred_mask.shape[0]):
        pred_masks_colored.append(mask_coloring(batch_pred_mask[i]))
    pred_masks_colored = np.concatenate(pred_masks_colored)
    pred_masks_colored = cv2.cvtColor(pred_masks_colored, cv2.COLOR_BGR2RGB)

    images = batch_x.reshape((batch_x.shape[0]*batch_x.shape[1], batch_x.shape[2], 3))
    vis_img = np.concatenate((images, masks_colored, pred_masks_colored), axis=1)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    return vis_img

