import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


# TODO add other loss compution jaccard


def compute_loss(prediction, target, bce_weight, metrics):
    # prediction[prediction > 1.0] = 1.0

    # prediction = torch.from_numpy(prediction)
    # target = torch.from_numpy(target)

    #print("pred", prediction.shape)
    #print("targ", target.shape)

    #print(prediction_.shape)
    #print(type(prediction_))

    #print(target)

    #pred_min = np.min(prediction.cpu().detach().numpy())
    #pred_max = np.max(prediction.cpu().detach().numpy())
    #pred_avg = np.average(prediction.cpu().detach().numpy())

    #print("pred", pred_min, pred_max, pred_avg)

    #targ_min = np.min(target.cpu().detach().numpy())
    #targ_max = np.max(target.cpu().detach().numpy())
    #targ_avg = np.average(target.cpu().detach().numpy())

    #print("targ", targ_min, targ_max, targ_avg)


    #bce = F.binary_cross_entropy_with_logits(prediction, target)
    #bce = torch.mean(torch.abs((1.0*target) - prediction))

    #tvesrky = tversky_loss(prediction, target, 0.5)

    #criterion = nn.CrossEntropyLoss()
    #ce = criterion(prediction_, prediction_)

    #loss = bce #* bce_weight # + tvesrky * (1 - bce_weight)
    #loss = ce

    #criterion = nn.CrossEntropyLoss(weight=bce_weight)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(prediction, target)

    #criterion = nn.BCELoss()
    #loss = criterion(prediction[:, 0, :, :], target)


    metrics["BCE"] += 0 #bce
    metrics["loss"] += loss
    metrics["tversky"] += 0 #tvesrky

    return loss

def print_metrics(metrics, samples , phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


# TODO variate epsilon
"""SOFT dice loss"""
"""def dice_loss( predicted, truth, epsilon):
    #predicted = torch.sigmoid(predicted)
    """
"""try:
        predicted = predicted.detach().numpy()
        truth = truth.detach().numpy()
    except TypeError:
        # Fix when we're running on CUDA
        predicted = predicted.cpu().detach().numpy()
        truth = truth.cpu().detach().numpy()
    """"""
    numerator = 2. * np.sum(predicted * truth)

    denominator = np.sum(predicted + truth)

    return 1 -( numerator +epsilon) / (denominator +epsilon)"""


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

    prediction = prediction.clone().detach()
    target = target.clone().detach()

    prediction = prediction.contiguous()
    target = target.contiguous()
    """try:
        pred = pred.detach().numpy()
        target = target.detach().numpy()
    except TypeError:
        # Fix when we're running on CUDA
        pred = pred.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

    """
    intersection = (prediction * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (prediction.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    print(loss.mean())
    return loss.mean()


if __name__ == '__main__':
    predicted = np.zeros((3, 3))
    predicted[1, 1] = 1

    truth = np.ones((3, 3))
    # truth[1, :] = 1

    # print(dice(predicted, truth, 1.))
    print(dice_loss(predicted, truth, 1.))
    print(tversky_loss(truth, predicted, 0.9))
