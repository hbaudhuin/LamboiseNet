import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import sklearn.metrics as skmetrics
import matplotlib.pyplot as plt

# TODO add other loss compution jaccard


def compute_loss(prediction, target, bce_weight, metrics):



    criterion = nn.CrossEntropyLoss(weight=bce_weight)
    loss = criterion(prediction, target)

    #criterion = nn.BCELoss()
    #loss = criterion(prediction[:, 0, :, :], target)


    metrics["loss"] += loss


    return loss

def print_metrics(metrics, samples , phase):
    outputs = []
    for k in metrics.keys():
        if type(metrics[k]) is not 'str' :
            print(str(k)+" "+str(metrics[k]))
        else :
            outputs.append("{}: {:4f}".format(k, metrics[k] / samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

def get_metrics(predicted, target, metrics_dict):

    predicted_copy = np.copy(predicted)

    for index, threshold in enumerate([ 0.1,0.2, 0.3,0.5, 0.7,0.9]):
        predicted_thresholded = predicted
        predicted_thresholded[predicted_copy >= threshold] = 1
        predicted_thresholded[predicted_copy < threshold] = 0
        #print("THRESHOLD CHANGE " +str(threshold))
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for i in range(len(predicted_thresholded)) :


            for j in range(len(predicted_thresholded[0])) :
                if predicted_thresholded[i, j] != target[i, j] :
                    if predicted_thresholded[i, j] == 0 :

                        false_negative += 1
                    else:

                        false_positive += 1
                elif predicted_thresholded[i, j] == 0 :

                    true_negative +=1
                else :

                    true_positive +=1


        if true_positive == 0 and false_negative == 0:
            recall= 1
        else:
            recall = true_positive / (true_positive + false_negative)

        if true_positive ==0 and false_positive == 0 :
            precision =1
        else :
            precision = true_positive / (true_positive+false_positive) #tpr

        if true_positive == 0 and false_negative == 0 :
            tpr = 1
        else :
            tpr = true_positive / (true_positive + false_negative)

        if true_negative == 0 and false_positive == 0 :
            specificity = 1
        else :
            specificity = true_negative / (true_negative + false_positive)

        if recall * precision == 0 :
            f1 = 0
        else :
            f1 = 2*recall*precision/(recall+precision)


        fpr = 1 - specificity#false_positive / (true_negative + false_positive)

        metrics_dict["F1"][index] += f1
        metrics_dict["Recall"][index] += recall
        metrics_dict["Precision"][index] += precision
        metrics_dict["TP"][index] += true_positive
        metrics_dict["TN"][index] += true_negative
        metrics_dict["FP"][index] += false_positive
        metrics_dict["FN"][index] += false_negative
        metrics_dict["TPR"][index] += tpr
        metrics_dict["FPR"][index] += fpr

    metrics_dict["AUC"] += (sum(metrics_dict["TPR"]) - sum(metrics_dict["FPR"]) +1) /2


def dice(prediction, target):
    smooth = 1.
    iflat = prediction  # .view(-1)
    tflat = target  # .view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


def tversky_loss(y_true, y_pred, beta):
    y_pred = y_pred.clone()
    y_true = y_true.clone()
    y_pred = torch.sigmoid(y_pred)
    try:
        y_pred = y_pred.detach().numpy()
        y_true = y_true.detach().numpy()
    except TypeError:
        # Fix when we're running on CUDA
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()

    numerator = np.sum(y_true * y_pred)

    denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)

    return 1 - (numerator + 1.) / (np.sum(denominator) + 1.)


def dice_loss(prediction, target):
    smooth = 1.
    beta = 0.5

    try:
        prediction = prediction.detach().numpy()
        target = target.detach().numpy()
    except TypeError:
        # Fix when we're running on CUDA
        prediction = prediction.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

    prediction= prediction[0]
    prediction[prediction > 0.5] = 1.0
    prediction[prediction <= 0.5] = 0.0
    numerator = np.sum(target * prediction)

    denominator = target * prediction + beta * (1 - target) * prediction + (1 - beta) * target * (1 - prediction)

    return 1 - (numerator + 1.) / (np.sum(denominator) + 1.)


if __name__ == '__main__':
    predicted = np.zeros((3, 3))
    predicted[1, 1] = 1

    truth = np.ones((3, 3))
    # truth[1, :] = 1
    n_thesholds = 6
    metrics = dict([("F1",np.zeros(n_thesholds)), ("Recall",np.zeros(n_thesholds)),
                            ("Precision",np.zeros(n_thesholds)), ("TP",np.zeros(n_thesholds)),
                            ("TN", np.zeros(n_thesholds)), ("FP",np.zeros(n_thesholds)), ("FN",np.zeros(n_thesholds)),
                            ("AUC", 0), ("TPR", np.zeros(n_thesholds)),
                            ("FPR", np.zeros(n_thesholds))])

    predicted = np.zeros((3, 3))
    predicted[1, 1] = 0.5
    predicted[0,1] = 0.25
    predicted[0,2] = 0.4

    target = np.zeros((3, 3))
    target[0,2] = 1
    get_metrics(predicted, target, metrics)

    print_metrics(metrics, 1, "test")

    dataset = np.zeros(1)

    plt.title('Receiver Operating Characteristic')
    plt.plot(metrics["FPR"] / len(dataset), metrics["TPR"] / len(dataset), 'b', label='AUC = %0.2f' % metrics["AUC"])
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





