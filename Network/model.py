import tensorflow as tf
from Network.backbone.architecture_backbones import backbone_model
from Network.head.archead import ArcHead


class MyModel(tf.keras.Model):
    def __init__(self, backbone, header):
        super(MyModel, self).__init__()
        self.backbone = backbone
        self.header = header

    def call(self, inputs, training=False):
        out = self.backbone(inputs, training=training)
        out = self.header(out)
        return out


if __name__ == '__main__':
    model = MyModel(backbone=backbone_model(type_model='Resnet_tf'),
                    header=ArcHead(num_classes=1000,
                                   kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.build(input_shape=(None, 160, 160, 3))
    print(model.summary())
