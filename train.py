"""
    Template to training
    Author: Nguyen Thai Hoc
    Date: 09-08-2022
"""
import tensorflow as tf
import argparse
from Network.model import MyModel
from Network.head.archead import ArcHead
from Tensorflow.TFRecord.tfrecord import TFRecordData
from training_supervisor import TrainingSupervisor
from LossFunction.losses import CosfaceLoss


def parser():
    args_parse = argparse.ArgumentParser(description="For training ...")
    args_parse.add_argument("--tfrecord_file", required=False,
                            type=str,
                            help="file dataset",
                            default=r"D:\hoc-nt\MFCosFace_Mlflow\Dataset\raw_tfrecords\lfw.tfrecords")
    args_parse.add_argument("--num_classes", required=False,
                            type=int,
                            help="the number of classes of dataset",
                            default=5749)
    args_parse.add_argument("--num_images", required=False,
                            type=int,
                            help="the number of images of dataset",
                            default=13233)
    args_parse.add_argument("--embedding_size", required=False,
                            type=int,
                            help="the number of images of dataset",
                            default=512)
    args_parse.add_argument("--batch_size", required=False,
                            type=int,
                            help="the amount of batch to training",
                            default=32)
    args_parse.add_argument("--epochs", required=False,
                            type=int,
                            help="the amount of batch to training",
                            default=10)
    args_parse.add_argument("--input_shape", required=False,
                            type=int,
                            help="the amount of batch to training",
                            default=160)
    args_parse.add_argument("--training_dir", required=False,
                            type=str,
                            help="The checkpoint training dir",
                            default=r"./Tensorboard")
    return args_parse.parse_args()


def run(**kwargs):
    print("Options:")
    for k, v in locals().items():
        print(f"  {k}: {v}")

    # get hyper-parameter
    tfrecord_file = kwargs['tfrecord_file']
    num_classes = kwargs['num_classes']
    num_images = kwargs['num_images']
    embedding_size = kwargs['embedding_size']
    batch_size = kwargs['batch_size']
    epochs = kwargs['epochs']
    input_shape = kwargs['input_shape']
    training_dir = kwargs['training_dir']

    # chosing model
    type_backbone = 'Resnet_tf'
    archead = ArcHead(num_classes=num_classes)
    model = MyModel(type_backbone='Resnet_tf',
                    input_shape=input_shape,
                    header=archead)
    model.build(input_shape=(None, input_shape, input_shape, 3))
    optimizer = tf.keras.optimizers.Adam(0.001, amsgrad=True, epsilon=0.001)
    model.summary()

    # init loss function
    loss_fn = CosfaceLoss(margin=0.5, scale=64, n_classes=num_classes)

    # dataloader
    dataloader_train = TFRecordData.load(record_name=tfrecord_file,
                                         shuffle=True,
                                         batch_size=batch_size,
                                         is_repeat=False,
                                         binary_img=True,
                                         is_crop=True,
                                         reprocess=False,
                                         num_classes=num_classes,
                                         buffer_size=2048)

    supervisor = TrainingSupervisor(train_dataloader=dataloader_train,
                                    validation_dataloader=None,
                                    optimizer=optimizer,
                                    loss_fn=loss_fn,
                                    model=model,
                                    save_freq=1000,
                                    monitor='categorical_accuracy',
                                    mode='max',
                                    training_dir=training_dir,
                                    name='Trainer_Supervisor')

    supervisor.restore(weights_only=False, from_scout=True)
    supervisor.train(epochs=epochs, steps_per_epoch=num_images // batch_size)


if __name__ == '__main__':
    args = parser()
    run(tfrecord_file=args.tfrecord_file,
        num_classes=args.num_classes,
        num_images=args.num_images,
        embedding_size=args.embedding_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        input_shape=args.input_shape,
        training_dir=args.training_dir)
