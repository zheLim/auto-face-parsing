import torch
SMOOTH = 1e-6


def iou(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum(
        (1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum(
        (1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (
                union + SMOOTH)  # We smooth our devision to avoid 0/0

    return iou.mean()  # Or thresholded.mean() if you are interested in average across the batch

