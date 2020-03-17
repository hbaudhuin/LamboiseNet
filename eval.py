import torch as torch
from tqdm import tqdm
from loss import dice_loss, tversky_loss, compute_loss, get_metrics, print_metrics
from image import save_masks
import numpy as np
from sklearn.metrics import auc, roc_curve
import torch.nn as nn
import logging
import matplotlib.pyplot as plt


def evaluation(model, dataset, device, save_mask=True, plot_roc=True, print_metric=True):
    # Set model modules to eval
    model.eval()
    loss = 0

    last_masks = [None] * len(dataset)
    last_truths = [None] * len(dataset)

    thesholds = [0, 0.0000001, 0.000001, 0.000005, 0.00001, 0.000025, 0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
    n_thesholds  = len(thesholds)

    to_plot_metrics = dict([("F1",np.zeros(n_thesholds)), ("Recall",np.zeros(n_thesholds)),
                            ("Precision",np.zeros(n_thesholds)), ("TP",np.zeros(n_thesholds)),
                            ("TN", np.zeros(n_thesholds)), ("FP",np.zeros(n_thesholds)), ("FN",np.zeros(n_thesholds)),
                            ("AUC", 0), ("TPR", np.zeros(n_thesholds)),
                            ("FPR", np.zeros(n_thesholds))])

    with tqdm(desc=f'Validation', unit='img') as progress_bar:
        for i, (image, ground_truth) in enumerate(dataset):

            image = image[0, ...]
            ground_truth = ground_truth[0, ...]

            last_truths[i] = ground_truth

            image = image.to(device)
            ground_truth = ground_truth.to(device)

            with torch.no_grad():
                mask_predicted = model(image)
            last_masks[i] = mask_predicted

            progress_bar.set_postfix(**{'loss': loss})
            bce_weight = torch.Tensor([1, 5]).to(device)
            loss += compute_loss(mask_predicted, ground_truth, bce_weight=bce_weight)

            get_metrics(mask_predicted[0, 0], ground_truth[0], to_plot_metrics, thesholds)

            progress_bar.update()

    if save_mask:
        save_masks(last_masks, last_truths, str(device), max_img=50, shuffle=False, color="red", filename="mask_predicted_test.png", threshold=thesholds[np.argmax(to_plot_metrics["F1"])])

    if print_metric:
        print_metrics(to_plot_metrics, len(dataset), "test set")


    # AVERAGING THE METRICS
    nb_images = len(dataset)
    for (k,v) in to_plot_metrics.items():
        to_plot_metrics[k] = v / nb_images

    # ROC
    if plot_roc:
        plt.title('Receiver Operating Characteristic')
        plt.plot(to_plot_metrics["FPR"], to_plot_metrics["TPR"], 'b',
                 label='AUC = %0.2f' % to_plot_metrics["AUC"])
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        #plt.show()
        plt.savefig("ROC.png")
        plt.show()
        plt.close("ROC.png")


    loss /= len(dataset)

    to_plot_metrics["loss"] = loss
    to_plot_metrics["best_threshold"] = thesholds[np.argmax(to_plot_metrics["F1"])]

    return to_plot_metrics
