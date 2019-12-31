import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from lib.dataset.dataset_utils import get_image_list
from lib.dataset.preprocess import get_augmentation


class HelenDataset(Dataset):
    def __init__(self, dataset_folder, policy, train=True):
        super().__init__()
        self.train = train
        if self.train:
            self.image_folder = os.path.join(dataset_folder, 'train')
        else:
            self.image_folder = os.path.join(dataset_folder, 'test')

        self.image_list = get_image_list(self.image_folder)
        self.preprocess_fn = get_augmentation(policy)

    def __getitem__(self, index):
        fname = self.image_list[index]
        image = cv2.imread(f'{fname}_image.jpg')
        mask = cv2.imread(f'{fname}_label.png', cv2.IMREAD_GRAYSCALE)

        image, mask = self.preprocess_fn(image, mask)

        image = image.transpose((2, 0, 1))
        mask = mask.astype(np.int64)
        return image, mask

    def __len__(self):
        return len(self.image_list)


def get_dataset(policy):
    preprocess_fn = get_augmentation(policy)
    return HelenDataset(
        '/home/administrator/dataset/helenstar_release',
        preprocess_fn)


if __name__ == '__main__':
    train_policy = {'OutputSize': (256, 256), 'Scale': {'disable': True},
                    'Rotation': {'disable': True}, 'Crop': {'disable': True},
                    'PaddingValue': 0}

    he_dataset = HelenDataset('/home/administrator/dataset/helenstar_release', train_policy)
    train_loader = torch.utils.data.DataLoader(he_dataset, batch_size=1)
    for img, msk in train_loader:
        pass
    cv2.waitKey(0)
    pass