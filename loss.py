import torch
import numpy as np
import torch.nn.functional as F

#TODO add other loss compution (jaccard, Tversky)



def compute_loss(prediction, target, bce_weight):
    #prediction[prediction > 1.0] = 1.0


    #prediction = torch.from_numpy(prediction)
    #target = torch.from_numpy(target)

    bce = F.binary_cross_entropy_with_logits(prediction, target)

    pred = torch.sigmoid(prediction)

    #dice = dice_loss(pred, target)

    loss = bce#dice#bce * bce_weight + dice * (1 - bce_weight)


    return loss


#TODO variate epsilon
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


def dice(input, target, smooth) :
    smooth = 1.
    iflat = input#.view(-1)
    tflat = target #.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


def tversky_loss(y_true, y_pred, beta):
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


def dice_loss(pred, target):
    smooth = 1.

    pred = pred.clone().detach()
    target = target.clone().detach()

    pred = pred.contiguous()
    target = target.contiguous()
    """try:
        pred = pred.detach().numpy()
        target = target.detach().numpy()
    except TypeError:
        # Fix when we're running on CUDA
        pred = pred.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

    """
    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    print(loss.mean())
    return loss.mean()


if __name__ == '__main__':
    predicted = np.zeros((3,3))
    predicted[1, 1]= 1

    truth = np.ones((3,3))
    #truth[1, :] = 1


    #print(dice(predicted, truth, 1.))
    print(dice_loss(predicted, truth, 1.))
    print(tversky_loss(truth, predicted, 0.9))