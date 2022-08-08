import os
import click
import mlflow
import tensorflow as tf
from utlis.argparser import parser_args
from Tensorflow.TFRecord.tfrecord import TFRecordData
from Tensorflow.Architecture.ModelFeatureExtraction.inception_resnet_v1 import InceptionResNetV1
from LossFunction.losses import CosfaceLoss


def set_env_vars():
    os.environ["MLFLOW_TRACKING_URI"] = "http://34.127.32.14:5000"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://34.127.32.14:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "admin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "hocmap123"


def train(run, model_name, mlflow_custom_log, **kwargs):
    args = parser_args()

    # get parameter
    path_file_tfrecords = args.file
    num_classes = args.num_classes
    num_images = args.num_images
    batch_size = args.batch_size
    embedding_size = args.embedding_size
    model_type = args.model_type
    learning_rate = args.lr

    # get data
    dataloader_train = TFRecordData.load(record_name=path_file_tfrecords,
                                         shuffle=True,
                                         batch_size=32,
                                         is_repeat=True,
                                         binary_img=True,
                                         is_crop=True,
                                         reprocess=False,
                                         num_classes=num_classes,
                                         buffer_size=10240)
    # build model
    model = InceptionResNetV1(num_classes=num_classes,
                              embedding_size=embedding_size,
                              model_type=model_type,
                              name="InceptionResNetV1")

    # compile model
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                        momentum=0.9, nesterov=True)
    model.compile(loss=CosfaceLoss(margin=0.5, scale=64, n_classes=num_classes),
                  optimizer=optimizer)

    # model fit
    model.fit(
        dataloader_train,
        validation_split=0.2,
        steps_per_epoch=num_images // batch_size,
        epochs=10,
    )

    if mlflow_custom_log:
        # write model summary
        summary = []
        model.summary(print_fn=summary.append)
        summary = "\n".join(summary)
        with open("model_summary.txt", "w") as f:
            f.write(summary)
        mlflow.log_artifact("model_summary.txt", artifact_path='model_summary')
        mlflow.tensorflow.autolog(model, "tf-model")

    # write model as json file
    with open("model.json", "w") as f:
        f.write(model.to_json())
    mlflow.log_artifact("model.json", artifact_path='model_summary')


# optional for mlflow
"""
    Mlflow settings parameters
"""


@click.command()
@click.option("--experiment-name",
              help="Experiment name",
              default="Test_exp_1", type=str)
@click.option("--mlflow-custom-log",
              help="Explicitly log params, metrics and model with mlflow.log_",
              default=True, type=bool)
@click.option("--tensorflow-autolog",
              help="Automatically log params, metrics and model with mlflow.tensorflow.autolog",
              default=True, type=bool)
@click.option("--mlflow-autolog",
              help="Automatically log params, metrics and model with mlflow.autolog",
              default=True, type=bool)
@click.option("--user",
              help="Automatically log params, metrics and model with mlflow.autolog",
              default='hoc-nt', type=str)
@click.option("--model-name",
              help="Registered model name",
              default='InceptionResNetV1', type=str)
def main(experiment_name, model_name, mlflow_autolog, tensorflow_autolog, mlflow_custom_log, user):
    print("Options:")
    for k, v in locals().items():
        print(f"  {k}: {v}")

    model_name = None if not model_name or model_name == "None" else model_name
    if not mlflow_autolog and not tensorflow_autolog:
        mlflow_custom_log = True

    if tensorflow_autolog:
        mlflow.tensorflow.autolog()
    if mlflow_autolog:
        mlflow.autolog()

    try:
        exp_id = mlflow.create_experiment(name=experiment_name)
    except Exception as e:
        exp_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id

    with mlflow.start_run(experiment_id=exp_id, run_name=model_name) as run:
        print("MLflow:")
        print("  run_id:", run.info.run_id)
        print("  experiment_id:", run.info.experiment_id)
        mlflow.set_tag("version.mlflow", mlflow.__version__)
        mlflow.set_tag("version.tensorflow", tf.__version__)
        mlflow.set_tag("mlflow_autolog", mlflow_autolog)
        mlflow.set_tag("tensorflow_autolog", tensorflow_autolog)
        mlflow.set_tag("mlflow_custom_log", mlflow_custom_log)
        mlflow.set_tag("Type of model", model_name)
        mlflow.set_tag("Developer", user)
        train(run, model_name, mlflow_custom_log)


if __name__ == '__main__':
    set_env_vars()
    main()
