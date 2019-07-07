# https://github.com/NVIDIA/DALI/blob/master/docs/examples/dataloading_tfrecord.ipynb
# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/load_data/tf_records.ipynb
import tensorflow as tf
import argparse
import torch
import numpy as np
import os
import cv2
from math import ceil
from torch.utils.data import Dataset

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def  image_example(image_string, mask_string):
    # input image must be in jpeg format
    image_shape = tf.image.decode_jpeg(image_string).shape
    mask_string = tf.image.decode_png(mask_string, channels=0) #  0: Use the number of channels in the PNG-encoded image.
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'mask_raw': _bytes_feature(mask_string),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Feature(feature=feature))

class CelebAMaskHQ(Dataset):
    def __init__(self, root):
        """
        The dataset is organized as
        CelebAMask-HQ /
            CelebA-HQ-img /
                0.jpg
                ...
            CelebAMask-HQ-mask-anno /
                00000_hair.png

        :param root:
        """
        self.mask_type = ['hair', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'l_lip', 'r_lip', 'mouth', 'neck', 'nose']
        self.img_indexs = list(range(30000)) # 30k images in dataset
        self.img_folder = os.path.join(root, 'CelebA-HQ-img')
        self.mask_folder = os.path.join(root, 'CelebAMask-HQ-mask-anno')
        # originally, annotation files are split into 15 folders.
        # Here we manually move all annotation files into CelebAMask-HQ-mask-anno folder.
        self.mask_filenames = set([x for x in os.listdir('CelebAMask-HQ-mask-anno') if 'png' in x])

    def __getitem__(self, idx):
        if idx not in self.img_indexs:
            raise IndexError('Index %i out of range 30000.'.format(idx))
        img_pth = '%s/%i.jpg'.format(self.img_folder, idx)
        image_string = open(img_pth, 'rb').read()

        mask = np.zeros((512, 512), dtype=np.uint8)
        for label_idx, m_name in enumerate(self.mask_type):
            filename = '%.5d_%s.png'.format(idx, m_name)
            # some part of annotations may not exist
            if filename in self.mask_filenames:
                # read gray image
                this_mask = cv2.imread(os.path.join(self.mask_folder, filename), 0)
                mask[this_mask != 0] = label_idx

        cv2.imencode('.png', mask)[1].tostring()
        return image_string, mask_string

    def __len__(self):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset_path', type=str, help='Path to celebAMask-HQ dataset root directory', required=True)
    parser.add_argument('--tfrecord_filename',  default='celebaMask',type=str)

    args = parser.parse_args()

    dataset = CelebAMaskHQ(args.dataset_path)

    training_set = []
    validation_set = []
    testing_set = []
    # train set: 000000-162769
    # valid set: 162770-182636
    # test  set: 182637-202598
    with open(os.path.join(args.dataset_path, 'CelebA-HQ-to-CelebA-mapping.txt')) as f:
        f.readline()
        for line in f.readlines():
            line = [x for x in line.split(' ') if x is not '']
            idx = int(line[0])
            orig_idx = int(line[1]) + 1 # idx is 0-based, but file
            if orig_idx < 162770:
                training_set.append(idx)
            elif orig_idx < 182637:
                validation_set.append(idx)
            else:
                testing_set.append(idx)
    pass

    train_shards = 10
    val_shards = 1
    test_shards = 1

    train_number_per_shard = int(ceil(len(training_set) / train_shards))
    val_number_per_shard = int(ceil(len(training_set) / val_shards))
    test_number_per_shard = int(ceil(len(training_set) / test_shards))

    n_shards = 0
    n_examples = 0
    n_train_exmaples = len(training_set)
    for shard_idx in range(train_shards):
        tfrecord_filename = '%s-%.2d-of-%.2d' % (args.tfrecord_filename, shard_idx, train_shards)
        with tf.python_io.TFRecordWriter(tfrecord_filename) as writer:
            for examples_idx in range(shard_idx * train_number_per_shard, (shard_idx + 1) * train_number_per_shard):
                if examples_idx > n_train_exmaples - 1:
                    break
                image_string, mask_string = dataset[training_set[examples_idx]]
                tf_example = image_example(image_string, mask_string)
                writer.write(tf_example.SerializeToString())