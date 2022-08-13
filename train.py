"""
    Template to training
    Author: Nguyen Thai Hoc
    Date: 09-08-2022
"""
import mlflow
import tensorflow as tf
import argparse
from Network.model import MyModel
from Network.head.archead import ArcHead
from Tensorflow.TFRecord.tfrecord import TFRecordData
from training_supervisor import TrainingSupervisor
from LossFunction.losses import CosfaceLoss
from evalute import EvaluteObjects
from config import train_config, mlflow_config
from utlis.utlis import set_env_vars


def parser_mlflow():
    return dict({
        'experiment_name': mlflow_config.experiment_name,
        'run_name_model_type': mlflow_config.run_name_model_type,
        'model_name': mlflow_config.model_name,
        'mlflow_autolog': mlflow_config.mlflow_autolog,
        'tensorflow_autolog': mlflow_config.tensorflow_autolog,
        'user_name': mlflow_config.user_name,
    })


def train(run, model_name, mlflow_custom_log):
    print("Options-Training:")
    for k, v in locals().items():
        print(f"  {k}: {v}")

    # get hyper-parameter
    tfrecord_file = train_config.tfrecord_file
    tfrecord_file_eval = train_config.tfrecord_file_eval
    file_pair_eval = train_config.file_pair_eval
    num_classes = train_config.num_classes
    num_images = train_config.num_images
    embedding_size = train_config.embedding_size
    batch_size = train_config.batch_size
    epochs = train_config.epochs
    input_shape = train_config.input_shape
    training_dir = train_config.training_dir
    export_dir = train_config.export_dir

    # chosing model
    type_backbone = 'Resnet_tf'
    archead = ArcHead(num_classes=num_classes)
    model = MyModel(type_backbone='Resnet_tf',
                    input_shape=input_shape,
                    embedding_size=embedding_size,
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
    supervisor.export(model=model.backbone, export_dir=export_dir)
    supervisor.mlflow_artifact(model=model.backbone)

    # evaluate ...
    eval_class = EvaluteObjects(tfrecord_file=tfrecord_file_eval,
                                file_pairs=file_pair_eval,
                                batch_size=batch_size)
    metrics = eval_class.activate(model=model.backbone, embedding_size=embedding_size)
    eval_class.mlflow_logs(dict_metrics=metrics)


def mlflow_run():
    args_mlflow = parser_mlflow()
    print("Options-Mlflow:")
    for k, v in args_mlflow:
        print(f"  {k}: {v}")

    args_mlflow['model_name'] = None if not args_mlflow['model_name'] or args_mlflow['model_name'] == "None" else \
        args_mlflow['model_name']
    if not args_mlflow['mlflow_autolog'] and not args_mlflow['tensorflow_autolog']:
        mlflow_custom_log = True

    if args_mlflow['tensorflow_autolog']:
        mlflow.tensorflow.autolog()
    if args_mlflow['mlflow_autolog']:
        mlflow.autolog()

    try:
        exp_id = mlflow.create_experiment(name=args_mlflow['experiment_name'])
    except Exception as e:
        exp_id = mlflow.get_experiment_by_name(name=args_mlflow['experiment_name']).experiment_id

    with mlflow.start_run(experiment_id=exp_id, run_name=args_mlflow['model_name']) as run:
        print("MLflow:")
        print("  run_id:", run.info.run_id)
        print("  experiment_id:", run.info.experiment_id)
        mlflow.set_tag("version.mlflow", mlflow.__version__)
        mlflow.set_tag("version.tensorflow", tf.__version__)
        mlflow.set_tag("mlflow_autolog", args_mlflow['mlflow_autolog'])
        mlflow.set_tag("tensorflow_autolog", args_mlflow['tensorflow_autolog'])
        mlflow.set_tag("mlflow_custom_log", mlflow_custom_log)
        mlflow.set_tag("Type of model", args_mlflow['model_name'])
        mlflow.set_tag("Developer", args_mlflow['user_name'])
        train(run, args_mlflow['model_name'], mlflow_custom_log)


if __name__ == '__main__':
    set_env_vars()
    mlflow_run()
