from pathlib import Path

projects_path = Path().parent.resolve()

tfrecord_file = projects_path / 'Dataset' / 'raw_tfrecords' / 'lfw.tfrecords'

tfrecord_file_eval = projects_path / 'Dataset' / 'raw_tfrecords' / 'lfw.tfrecords'

file_pair_eval = projects_path / 'Dataset' / 'raw_tfrecords' / 'lfw.txt'

num_classes = 5749

num_images = 13233

embedding_size = 512

batch_size = 32

epochs = 10

input_shape = 160

training_dir = r'./Tensorboard'

export_dir = r'./Model'
