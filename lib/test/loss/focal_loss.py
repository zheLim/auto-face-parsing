import tensorflow as tf
import numpy as np
from lib.loss.loss import FocalLoss


def test_focal_loss():
    logits = tf.constant([[[[0.9, 0., 0.], [0.1, 0.8, 0.1], [0.1, 0., 0.9]],
                           [[0.1, 0., 0.9], [0.1, 0.8, 0.1], [0.1, 0., 0.9]],
                           [[0.1, 0., 0.9], [0.1, 0.8, 0.1], [0.1, 0., 0.9]]],
                          [[[0.9, 0., 0.], [0.1, 0.8, 0.1], [0.1, 0., 0.9]],
                           [[0.1, 0., 0.9], [0.1, 0.8, 0.1], [0.1, 0., 0.9]],
                           [[0.1, 0., 0.9], [0.1, 0.8, 0.1], [0.1, 0., 0.9]]]
                          ])
    # [[0.9, 0.1, 0.1],
    #  [0.1, 0.1, 0.1],
    #  [0.1, 0.1, 0.1]],
    #
    # [[0. , 0.8, 0. ],
    #   [0. , 0.8, 0. ],
    #   [0. , 0.8, 0. ]],
    #
    # [[0. , 0.1, 0.9],
    #  [0.9, 0.1, 0.9],
    #  [0.9, 0.1, 0.9]]
    labels = tf.constant([[[0, 1, 2],
                           [2, 1, 2],
                           [2, 1, 2]],
                          [[0, 1, 2],
                           [2, 1, 2],
                           [2, 1, 2]]
                          ])

    floss = FocalLoss(3, ohem_thresh=1., min_batch_size=1)
    res = floss(logits, labels)
    print(res)
    assert np.allclose(res.numpy(), 1.1713442)


if __name__ == '__main__':
    test_focal_loss()