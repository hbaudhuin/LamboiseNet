import torch as torch
from tqdm import tqdm
from loss import dice_loss, tversky_loss, compute_loss, get_metrics, print_metrics
from image import save_masks
import numpy as np
from sklearn.metrics import auc, roc_curve
import torch.nn as nn
import logging
import matplotlib.pyplot as plt


def evaluation(model, dataset, device, metrics):
    # Set model modules to eval
    model.eval()
    loss = 0

    last_masks = [None] * len(dataset)
    last_truths = [None] * len(dataset)

    n_thesholds  = 6

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
            bce_weight = torch.Tensor([0.1, 0.9]).to(device)
            loss += compute_loss(mask_predicted, ground_truth, bce_weight=bce_weight, metrics=metrics)

            get_metrics(mask_predicted[0, 0], ground_truth[0], to_plot_metrics)

            progress_bar.update()

    save_masks(last_masks, last_truths, str(device), max_img=50, shuffle=False, color="red", filename="mask_predicted_test.png")

    print_metrics(to_plot_metrics, len(dataset), "test set")

    #ROC

    plt.title('Receiver Operating Characteristic')
    plt.plot(to_plot_metrics["FPR"] / len(dataset), to_plot_metrics["TPR"] / len(dataset), 'b',
             label='AUC = %0.2f' % to_plot_metrics["AUC"])
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plt.savefig("ROC.png")
    plt.show()
    plt.close("ROC.png")
    #AUC


    loss /= len(dataset)

    return loss
