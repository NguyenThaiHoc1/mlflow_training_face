import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import Sequential
from tensorflow.keras.applications import resnet50, vgg16, imagenet_utils
from Network.backbone.resnet import ResNet_v1_101, ResNet_v1_34


def backbone_model(type_model,
                   input_shape=None,
                   embedding_size=512):
    backbone = None

    if type_model == 'ResNet_v1_101':
        input_tensor = tf.keras.layers.Input(shape=(input_shape, input_shape, 3))
        output = ResNet_v1_101(include_top=False)(input_tensor)
        backbone = tf.keras.Model(input_tensor, output)
    elif type_model == 'ResNet_v1_34':
        input_tensor = tf.keras.layers.Input(shape=(None, None, 3))
        output = ResNet_v1_34(include_top=False)(input_tensor)
        backbone = tf.keras.Model(input_tensor, output)
    elif type_model == 'Resnet_tf':
        backbone = resnet50.ResNet50(include_top=False, input_shape=(input_shape, input_shape, 3))
        output = tf.keras.layers.Flatten()(backbone.output)
        backbone = tf.keras.Model(backbone.input, output)
    elif type_model == 'Vgg16':
        backbone = vgg16.VGG16(include_top=False, input_shape=(input_shape, input_shape, 3))
        output = tf.keras.layers.Flatten()(backbone.output)
        backbone = tf.keras.Model(backbone.input, output)

    output = tf.keras.layers.Dense(embedding_size)(backbone.output)
    assert backbone is not None, "Please checking in backbone creator."
    return tf.keras.Model(backbone.input, output)


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
