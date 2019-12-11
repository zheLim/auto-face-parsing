import cv2
import tensorflow as tf
import tensorboard
import os
import yaml
from lib.model.unet import Unet
from lib.model.seg_hrnet import HighResolutionNet
from lib.dataset.celebAMaskHQ_tf import get_tf_dataset
from lib.loss.focal_loss import FocalLoss, OhemLoss
from lib.utils.visualization import visual_image_and_segmentation


def main(params):
    save_dir = 'outputs'
    for kind in ['image', 'model', 'log']:
        this_dir = os.path.join(save_dir, kind)
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    n_classes = 19
    with open('config/hrnet.yaml') as f:
        config_dict = yaml.load(f)
    model = HighResolutionNet(config_dict['MODEL'])

    tf_folder = '/workspace/cpfs-data/dataset/celebAMask-HQ-tfrecord/train/tfrecord/'
    train_policy = {'OutputSize': (256, 256), 'Scale': {'disable': True},
                    'Rotation': {'disable': True}, 'Crop': {'disable': True}, 'PaddingValue': 0}
    train_dataset = get_tf_dataset(tf_folder, train_policy, shuffle=True, batch_size=16)
    valid_policy = {'OutputSize': (256, 256), 'Scale': {'disable': True},
                    'Rotation': {'disable': True}, 'Crop': {'disable': True}, 'PaddingValue': 0}
    valid_dataset = get_tf_dataset(tf_folder, valid_policy, shuffle=False, batch_size=16)

    #loss_scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='sparse_categorical_crossentropy')
    #loss_focal = FocalLoss(n_classes, epsilon=1e-7, gamma=2.0, ohem_thresh=5, min_batch_size=16)
    ohem_loss = OhemLoss(n_classes, ohem_thresh=0.95, batch_size=16, width=256, min_keep=None, epsilon=1e-7, gamma=2.0)
    metric_mean_iou = tf.keras.metrics.MeanIoU(num_classes=n_classes)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    writer = tf.summary.create_file_writer(f'{save_dir}/log')

    tf.keras.backend.set_learning_phase(True)
    iter = 0
    with writer.as_default():
        for epoch in range(100):
            for (x_batch_train, y_batch_train) in train_dataset:
                iter += 1
                with tf.GradientTape() as tape:
                    predict_logits = model(x_batch_train)
                    losses = ohem_loss(y_batch_train, predict_logits)
                grads = tape.gradient(losses, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                if iter % 200 == 0:
                    print('Training loss at iteration %s: %s' % (iter, float(losses)))
                    vis_img_train = visual_image_and_segmentation(x_batch_train, y_batch_train, tf.math.argmax(predict_logits, axis=3))
                    cv2.imwrite(f'{save_dir}/image/train_{iter}.jpg', vis_img_train)
                    tf.summary.scalar('train loss', float(losses), step=iter)

                if iter % 1000 == 0:
                    tf.keras.backend.set_learning_phase(False)
                    for (x_batch_valid, y_batch_valid) in valid_dataset:
                        predict_logits = model(x_batch_valid)
                        predict_mask = tf.math.argmax(predict_logits, axis=3)
                        metric_mean_iou(y_batch_valid, predict_mask)
                    vis_img_valid = visual_image_and_segmentation(x_batch_valid, y_batch_valid, predict_mask)
                    cv2.imwrite(f'{save_dir}/image/valid_{iter}.jpg', vis_img_valid)
                    batch_mean_iou = metric_mean_iou.result()
                    print('Validation acc : %s' % (float(batch_mean_iou),))
                    tf.summary.scalar('Validation accuracy', float(batch_mean_iou), step=iter)
                    writer.flush()
                    # Reset training metrics at the end of each epoch
                    metric_mean_iou.reset_states()
                    model.save_weights(f'{save_dir}/model/{iter}')
                    tf.keras.backend.set_learning_phase(True)


if __name__ == '__main__':
    params = None
    main(params)