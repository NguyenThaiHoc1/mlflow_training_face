import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.applications import resnet50, vgg16
from Network.backbone.resnet import ResNet_v1_101, ResNet_v1_34


def backbone_model(type_model, embedding_size=512):
    backbone = None
    if type_model == 'ResNet_v1_101':
        backbone = ResNet_v1_101(include_top=False)
    elif type_model == 'ResNet_v1_34':
        backbone = ResNet_v1_34(include_top=False)
    elif type_model == 'Resnet_tf':
        backbone = resnet50.ResNet50(include_top=False)
    elif type_model == 'Vgg16':
        backbone = vgg16.VGG16(include_top=False)

    output = backbone.output
    output = tf.keras.layers.Dense(embedding_size)(output)
    assert backbone is not None, "Please checking in backbone creator."
    return tf.keras.Model(backbone.input, output, name=type_model)


if __name__ == '__main__':
    output_based_network = backbone_model(type_model='Resnet_tf')

    # model = MyModel(output_based_network)
    # model.build(input_shape=(None, 160, 160, 3))
    # print(model.summary())

    model = Sequential([
        tf.keras.Input(shape=(112, 112, 3)),
        output_based_network,
        # tf.keras.layers.Dense(512),
    ])
    print(model.summary())
