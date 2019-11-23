import tensorflow as tf


class FocalLoss(object):
    def __init__(self, num_classes, epsilon=1e-7, gamma=2.0, ohem_thresh=0.5, min_batch_size=8):
        self.num_classes = tf.constant(num_classes, tf.int32)
        self.epsilon = tf.constant(epsilon, tf.float32)
        self.gamma = tf.constant(gamma, tf.float32)
        self.ohem_thresh = tf.constant(ohem_thresh, tf.float32)
        self.n_min = tf.constant(min_batch_size, tf.int32)

    def __call__(self, labels, logits):

        probability = tf.nn.softmax(logits)
        one_hot_labels = tf.one_hot(labels, self.num_classes)
        clipped_prob = tf.clip_by_value(probability, self.epsilon, 1-self.epsilon)
        ce = tf.multiply(one_hot_labels, -tf.math.log(clipped_prob))
        factor = tf.pow(tf.subtract(1, probability), self.gamma)
        floss = tf.reduce_sum(tf.multiply(factor, ce), axis=(1, 2, 3))

        floss = self.online_hard_example_mining(floss)
        return tf.reduce_mean(floss)

    @tf.function
    def online_hard_example_mining(self, losses):
        losses_sorted = tf.sort(losses, direction='DESCENDING')
        if losses_sorted[self.n_min] < self.ohem_thresh:
            return losses_sorted[self.n_min]
        return losses_sorted[losses_sorted > self.ohem_thresh]


class OhemLoss(object):
    def __init__(self, num_classes, ohem_thresh=0.9, batch_size=None, width=None, min_keep=None, epsilon=1e-7, gamma=2.0):
        self.num_classes = tf.constant(num_classes, tf.int32)
        self.epsilon = tf.constant(epsilon, tf.float32)
        self.ohem_thresh = -tf.math.log(tf.constant(ohem_thresh, tf.float32))
        if batch_size is not None and width is not None:
            self.min_keep = tf.constant(width ** 2 // 16, tf.int32)
        elif min_keep is not None:
            if isinstance(min_keep, tf.Tensor):
                self.min_keep = min_keep
            else:
                self.min_keep = tf.constant(min_keep)
        else:
            raise ValueError('Batch size and width or min_keep shall be not none!')
        self.ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    def __call__(self, labels, logits):
        ohem_loss = self.ce_loss(labels, logits)

        floss = self.online_hard_example_mining(ohem_loss)
        return tf.reduce_mean(floss)

    def online_hard_example_mining(self, losses):
        losses_sorted = tf.sort(tf.reshape(losses, [-1]), direction='DESCENDING')
        if losses_sorted[self.min_keep] < self.ohem_thresh:
            return losses_sorted[self.min_keep]
        return losses_sorted[losses_sorted > self.ohem_thresh]