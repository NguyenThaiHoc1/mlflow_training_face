import tensorflow as tf
from Network.backbone.architecture_backbones import backbone_model
from Network.head.archead import ArcHead


class MyModel(tf.keras.Model):
    def __init__(self, type_backbone, header, input_shape, embedding_size):
        super(MyModel, self).__init__()
        self.backbone = backbone_model(type_model=type_backbone,
                                       input_shape=input_shape, embedding_size=embedding_size)
        self.header = header

    def call(self, inputs, training=False):
        out = self.backbone(inputs, training=training)
        out = self.header(out)
        return out


if __name__ == '__main__':
    """
        - ResNet_v1_101
        - ResNet_v1_34
        - Resnet_tf
        - Vgg16
    """
    input_shape = 250
    model = MyModel(type_backbone='Resnet_tf',
                    input_shape=input_shape,
                    header=ArcHead(num_classes=1000, kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.build(input_shape=(None, input_shape, input_shape, 3))
    print(model.summary())

    x = tf.keras.layers.Input(shape=(input_shape, input_shape, 3))
    out = model(x, training=True)

    print(f"input: {x}")
    print(f"output: {out}")
    print("DONE ...")
