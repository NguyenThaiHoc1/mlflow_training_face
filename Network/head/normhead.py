import tensorflow as tf
from tensorflow.keras.layers import Dense


def regular(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)


class NormHead(tf.keras.layers.Layer):

    def __init__(self, name, num_classes):
        super(NormHead, self).__init__(name=name)

        self.dense = Dense(num_classes, kernel_regularizer=regular(weights_decay=5e-4), name='NormHead_FC_Dense')

    def call(self, inputs):
        out = self.dense(inputs)
        return out
