from collections import defaultdict
import os
from Tensorflow.TFRecord.tfrecord import TFRecordData
from utlis.evalute import *
from functools import partial
from matplotlib import pyplot as plt
import seaborn as sb


class EvaluteObjects(object):

    def __init__(self, tfrecord_file, file_pairs, batch_size=32):
        self.tfrecord_file = tfrecord_file
        self.file_pairs = file_pairs
        self.batch_size = batch_size

        # function call
        assert self.tfrecord_file is not None, "Please provide dataset to evalute."
        assert self.file_pairs is not None, "Please provide file_pairs to evalute."
        self._parser_dataloader()
        self._read_pair()
        self._read_actual_issame()

    def _read_pair(self):
        self.pairs = read_pairs(pairs_filename=self.file_pairs)

    def _parser_dataloader(self):
        self.dataloader = TFRecordData.load(record_name=self.tfrecord_file,
                                            shuffle=False,
                                            batch_size=self.batch_size,
                                            is_repeat=False,
                                            binary_img=True,
                                            is_crop=False,
                                            reprocess=True,
                                            num_classes=None,
                                            buffer_size=2048)

    def _read_actual_issame(self):
        dict_person = defaultdict(partial(defaultdict, list))
        for data, filenames in self.dataloader:
            for ndarray_image, filename in zip(data, filenames):
                name = str(bytes.decode(filename.numpy()))
                parent_path = os.path.dirname(name)
                base_name = os.path.basename(name).split('.')[0]  # elimate extension
                dict_person[parent_path].update({base_name: ndarray_image})

        self.list_arrays, self.actual_issame = get_paths(dict_person, self.pairs)
        del dict_person

    def activate(self, model, embedding_size, plot=True):
        assert self.list_arrays is not None or self.actual_issame is not None, "Please check init state."
        distances, labels = evalute(embedding_size=embedding_size, step=32,
                                    model=model, carray=self.list_arrays, issame=self.actual_issame)
        metrics = evaluate_lfw(distances=distances, labels=labels)

        metrics_dict = {
            'mean_accuracy': np.mean(metrics['accuracy']),
            'std_accuracy': np.std(metrics['accuracy']),
            'mean_precision': np.mean(metrics['precision']),
            'std_precision': np.std(metrics['precision']),
            'mean_recall': np.mean(metrics['recall']),
            'std_recall': np.std(metrics['recall']),
            'mean_best_distances': np.mean(metrics['best_distances']),
            'std_best_distances': np.std(metrics['best_distances']),
            'mean_tar': np.mean(metrics['tar']),
            'std_tar': np.std(metrics['tar']),
            'mean_far': np.mean(metrics['far']),
            'roc_auc': metrics['accuracy'],
        }

        print(metrics_dict)

        text_metrics = "Accuracy on dataset: {:.4f}+-{:.4f}\nPrecision {:.4f}+-{:.4f}\nRecall {:.4f}+-{:.4f}" \
                       "\nBest distance threshold: {:.2f}+-{:.2f}" \
                       "\nTAR: {:.4f}+-{:.4f} @ FAR: {:.4f}".format(metrics_dict['mean_accuracy'],
                                                                    metrics_dict['std_accuracy'],
                                                                    metrics_dict['mean_precision'],
                                                                    metrics_dict['std_precision'],
                                                                    metrics_dict['mean_recall'],
                                                                    metrics_dict['std_recall'],
                                                                    metrics_dict['mean_best_distances'],
                                                                    metrics_dict['std_best_distances'],
                                                                    metrics_dict['mean_tar'],
                                                                    metrics_dict['std_tar'],
                                                                    metrics_dict['mean_far'])  # roc auc bi sai

        if plot:
            title = 'All metrics'
            fig, axes = plt.subplots(1, 2)
            fig.suptitle(title, fontsize=15)
            fig.set_size_inches(14, 6)
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            axes[0].set_title('distance histogram')
            sb.distplot(distances[labels == True], ax=axes[0], label='distance-true')
            sb.distplot(distances[labels == False], ax=axes[0], label='distance-false')
            axes[0].legend()

            axes[1].text(0.05, 0.3, text_metrics, fontsize=20)
            axes[1].set_axis_off()
            plt.savefig("plot_all_metrics.png")

        return metrics_dict


# if __name__ == '__main__':
#     from collections import defaultdict
#     import os
#
#     eval_class = EvaluteObjects(tfrecord_file=r'D:\hoc-nt\MFCosFace_Mlflow\Dataset\raw_tfrecords\lfw.tfrecords',
#                                 file_pairs=r'D:\hoc-nt\MFCosFace_Mlflow\Dataset\pairs\lfw_pairs.txt')
#
#     # loading model checkpoint
#     model = None
#
#     # activate
#     # eval_class.activate(model, embedding_size=512)
