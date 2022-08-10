import tensorflow as tf


class ArcHead(tf.keras.layers.Layer):

    def __init__(self, num_classes, kernel_regularizer=tf.keras.regularizers.l2(5e-4), **kwargs):
        super(ArcHead, self).__init__(**kwargs)
        self.units = num_classes
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[-1], self.units),
                                 dtype=tf.float32,
                                 initializer=tf.keras.initializers.HeNormal(),
                                 regularizer=self.kernel_regularizer,
                                 trainable=True)

        self.built = True

    @tf.function
    def call(self, inputs):
        # Step 1: l2_norm input
        inputs = tf.nn.l2_normalize(inputs, axis=1)

        # Step 2: l2_norm weight
        weight = tf.nn.l2_normalize(self.W, axis=0)

        return tf.matmul(inputs, weight)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units,
                       "kernel_regularizer": self.kernel_regularizer})
        return config
