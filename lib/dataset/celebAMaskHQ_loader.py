from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec
from lib.utils.visualization import mask_coloring


class celebAMaskHQPipeline(Pipeline):
    """Forget this, Nvidia dali cannot build pipeline for segmentation task. """
    def __init__(self, tfrecord_list, tfrecord_indexes, batch_size, num_threads, device_id):
        super(celebAMaskHQPipeline, self).__init__(batch_size,
                                               num_threads,
                                               device_id)
        self.input = ops.TFRecordReader(path=tfrecord_list,
                                        index_path=tfrecord_indexes,
                                        features={"height": tfrec.FixedLenFeature((), tfrec.int64, -1),
                                                  'width': tfrec.FixedLenFeature([1], tfrec.int64, -1),
                                                  'mask_raw': tfrec.FixedLenFeature([],  tfrec.string, ''),
                                                  'image_raw': tfrec.FixedLenFeature([], tfrec.string, ''),
                                                  })
        self.image_decode = ops.HostDecoder(device="cpu", output_type=types.RGB)
        self.mask_decode = ops.HostDecoder(output_type=types.GRAY)
        self.im_resize = ops.Resize(device="cpu", resize_shorter=512.)
        self.mask_resize = ops.Resize(device="cpu", resize_shorter=512., interp_type=types.DALIInterpType.INTERP_NN)
        self.cmnp = ops.CropMirrorNormalize(device="cpu",
                                            output_dtype=types.FLOAT,
                                            crop=(256, 256),
                                            mean=[0, 0, 0],
                                            std=[1, 1, 1])
        self.crop = ops.Crop(crop=(256, 256))
        self.uniform = ops.Uniform(range = (0.0, 1.0))

        self.iter = 0

    def define_graph(self):
        inputs = self.input()
        image = self.image_decode(inputs['image_raw'])
        mask = self.mask_decode(inputs['mask_raw'])
        image = self.im_resize(image)
        mask = self.mask_resize(mask)
        # image, mask = self.crop([image, mask], crop_pos_x = self.uniform(),
        #                    crop_pos_y = self.uniform())

        # image, mask = self.cmnp([image, mask], crop_pos_x = self.uniform(),
        #                    crop_pos_y = self.uniform())
        return (image, mask)

    def iter_setup(self):
        pass

def show_images(image_batch, masks, batch_size):
    columns = 4 * 2
    rows = (batch_size + 1) // (columns)*2
    fig = plt.figure(figsize = (32,(32 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(0, rows*columns, 2):
        img_chw = image_batch.at(int(j/2))
        img_hwc = np.transpose(img_chw, (1, 2, 0)) * 127.5 + 127.5
        #img_hwc = (np.transpose(img_chw, (1, 2, 0))*0.5 + 0.5) * 255

        mask = masks.at(int(j/2))
        mask = mask_coloring(mask)
        plt.subplot(gs[j])
        plt.axis("off")
        plt.imshow(img_hwc.astype(np.uint8))
        plt.subplot(gs[j+1])
        plt.imshow(mask)

    plt.savefig('../../temp/data_test.jpg')


