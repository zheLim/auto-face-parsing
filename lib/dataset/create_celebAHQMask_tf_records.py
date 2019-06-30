# https://github.com/NVIDIA/DALI/blob/master/docs/examples/dataloading_tfrecord.ipynb
# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/load_data/tf_records.ipynb
import tensorflow as tf
import argparse

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--tf_record_filename', type=str)

    args = parser.parse_args()
    with tf.python_io.TFRecordWriter(args.tf_record_filename) as writer:
        for filename, label in image_labels.items():
            image_string = open(filename, 'rb').read()
            tf_example = image_example(image_string, label)
            writer.write(tf_example.SerializeToString())

    # should use DALI/tools/tfrecord2idx.