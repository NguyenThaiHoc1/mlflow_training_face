import argparse


def parser_args():
    arg_parser = argparse.ArgumentParser(description="Show image raw")
    arg_parser.add_argument('--file',
                            default=r'D:\hoc-nt\MFCosFace_Mlflow\Dataset\raw_tfrecords\lfw_new_mask.tfrecords',
                            type=str,
                            required=False,
                            help='This is link which thought to tfreocrds file')
    arg_parser.add_argument('--num_classes',
                            default=8335,
                            type=int,
                            required=False,
                            help='The number of classes of dataset link')
    arg_parser.add_argument('--num_images',
                            default=666750,
                            type=int,
                            required=False,
                            help='The amount of images of dataset link')
    arg_parser.add_argument('--batch_size',
                            default=32,
                            type=int,
                            required=False,
                            help='batch_size when dataloader take')
    arg_parser.add_argument('--embedding_size',
                            default=512,
                            type=int,
                            required=False,
                            help='Length of feature vector')
    arg_parser.add_argument('--model_type',
                            default='ArcHead',
                            type=str,
                            required=False,
                            help='Model type head of total model')
    arg_parser.add_argument('--lr',
                            default=1e-3,
                            required=False,
                            help='Learning rate of model')
    return arg_parser.parse_args()
