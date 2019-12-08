import torch
import numpy as np
import torch as tf
import tensorflow as tensor

#TODO add other loss compution (jaccard, Tversky)


#TODO variate epsilon
"""SOFT dice loss"""
def dice_loss( predicted, truth, epsilon):
    #predicted = torch.sigmoid(predicted)
    """try:
        predicted = predicted.detach().numpy()
        truth = truth.detach().numpy()
    except TypeError:
        # Fix when we're running on CUDA
        predicted = predicted.cpu().detach().numpy()
        truth = truth.cpu().detach().numpy()
    """
    numerator = 2. * np.sum(predicted * truth)

    denominator = np.sum(predicted + truth)

    return 1 -( numerator +epsilon) / (denominator +epsilon)


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


if __name__ == '__main__':
    predicted = np.zeros((3,3))
    predicted[1, 1]= 1

    truth = np.ones((3,3))
    #truth[1, :] = 1


    #print(dice(predicted, truth, 1.))
    print(dice_loss(predicted, truth, 1.))
    print(tversky_loss(truth, predicted, 0.9))