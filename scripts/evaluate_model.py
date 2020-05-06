import os
import numpy as np
import torch
import argparse
from scipy import interp
from matplotlib import pyplot as plt
from itertools import cycle
import seaborn as sns
from sklearn import metrics
from torch.utils.data import DataLoader

import config_file
from extract_features import split_spectrogram
from audio_tagging_pytorch.datasets.mtt import MTTDataset
from audio_tagging_pytorch.models.musicnn import Musicnn


def calculate_auc(predictions, ground_truth):
    y_pred = []
    y_true = []
    for file_id in ground_truth.keys():
        avg = np.mean(predictions[file_id], axis=0)
        y_pred.append(avg)
        y_true.append(ground_truth[file_id])

    print('Predictions are averaged, now computing AUC..')
    roc_auc, pr_auc, roc_auc_points, tpr, fpr = compute_auc(y_true, y_pred)
    return np.mean(roc_auc), np.mean(pr_auc), roc_auc_points, tpr, fpr


def compute_auc(y_true, y_pred):
    pr_auc = []
    roc_auc = []
    estimated = np.reshape(y_pred, (len(y_pred), 50))
    true = np.reshape(y_true, (len(y_true), 50))
    n_classes = estimated.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc_points = dict()
    for count in range(n_classes):
        fpr[count], tpr[count], _ = metrics.roc_curve(true[:, count],
                                                      estimated[:, count])
        roc_auc_points[count] = metrics.auc(fpr[count], tpr[count])
        fpr["micro"], tpr["micro"], _ = metrics.roc_curve(
            true.ravel(), estimated.ravel())
        roc_auc_points["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
        try:
            roc_metric = metrics.roc_auc_score(true[:, count],
                                               estimated[:, count])
            pr_auc_metric = metrics.average_precision_score(
                true[:, count], estimated[:, count])
        except:
            print("roc not computed")
        roc_auc.append(roc_metric)
        pr_auc.append(pr_auc_metric)
    return roc_auc, pr_auc, roc_auc_points, tpr, fpr


def plot_roc(roc_auc_points, tpr, fpr, n_classes):
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc_points["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    sns.set_style("white")
    plt.plot(fpr["micro"],
             tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
             ''.format(roc_auc_points["micro"]),
             linestyle=':',
             linewidth=4)

    plt.plot(fpr["macro"],
             tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
             ''.format(roc_auc_points["macro"]),
             linestyle=':',
             linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    # plt.savefig(config_file.DATA_PATH + "roc_mode3.png", tpi=400)
    plt.show()


def evaluate(test_dataset, model_number):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    cuda_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_loader = DataLoader(test_dataset, TEST_BATCH_SIZE)
    if model_number == "2":
        path_to_model = config_file.TEMP_POOLING_MODEL
        model = Musicnn(y_input_dim=96,
                        filter_type="timbral",
                        k_height_factor=0.7,
                        k_width_factor=1.,
                        filter_factor=1.6,
                        pool_type="temporal")
    elif model_number == "3":
        path_to_model = config_file.ATTENTION_MODEL
        model = Musicnn(y_input_dim=96,
                        filter_type="timbral",
                        k_height_factor=0.7,
                        k_width_factor=1.,
                        filter_factor=1.6,
                        pool_type="attention")
    print(path_to_model)

    model.eval()
    checkpoint = torch.load(path_to_model, map_location="cuda:0")
    model.load_state_dict(checkpoint['state_dict'])
    model.to(cuda_device)
    model = model.double()

    predictions = {}
    ground_truth = {}

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            features, labels, file_id = batch
            labels = labels.float()
            # TODO split spectrogram should be part of MTTDataset when no random sampling
            X = split_spectrogram(torch.squeeze(features.T))
            # X = features.to(device=cuda_device)
            y = labels.to(device=cuda_device)
            X = torch.tensor(X)
            X = X.to(device=cuda_device)

            for input_mel in X:
                input_mel = input_mel.T.view(1, 1, 96, 187)
                output = torch.squeeze(model(input_mel))
                id_key = int(file_id[0])
                if id_key not in ground_truth.keys():
                    ground_truth[id_key] = torch.Tensor.cpu(y).numpy()
                    if 50 - np.count_nonzero(
                            torch.Tensor.cpu(y).numpy()) == 50:
                        print(id_key)
                    predictions[id_key] = [torch.Tensor.cpu(output).numpy()]
                else:
                    predictions[id_key].append(
                        [torch.Tensor.cpu(output).numpy()])
            print("batch", i, "of", len(test_loader))

    return predictions, ground_truth


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute mel spectrogram of input audio")
    parser.add_argument("model_number",
                        type=str,
                        help="number of pretrained model to load - 2 is best")
    args = parser.parse_args()

    model_number = args.model_number

    config = config_file.config_training
    TEST_BATCH_SIZE = 1
    FILE_INDEX = config_file.DATASET + config['index_file']
    FILE_GROUND_TRUTH_TEST = config_file.DATASET + config['gt_test']

    test_dataset = MTTDataset(config_file.DATA_PATH,
                              config["index_file"],
                              config["gt_test"],
                              187,
                              random_sampling=False)
    predictions, ground_truth = evaluate(test_dataset, model_number)
    mean_roc, mean_auc, roc_auc_points, tpr, fpr = calculate_auc(
        predictions, ground_truth)
    print(mean_roc, mean_auc)
    plot_roc(roc_auc_points, tpr, fpr, n_classes=50)
