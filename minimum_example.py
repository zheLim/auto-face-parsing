import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras


class Model(keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        self.conv_list = [[
                          None, 
                          ]]

    def call(self, x):
        for conv_layer in self.conv_list:
            conv_layer = self.conv_list[0]
            for c in conv_layer:
                if c is not None:
                    x = c(x)
        return x


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    model = Model()
    image = tf.constant(np.ones((1, 256, 256, 3), dtype=np.float32))
    out = model(image)

    model.save_weights('100')