import torch
import numpy as np

#TODO add other loss compution (jaccard, Tversky)


#TODO variate epsilon
"""SOFT dice loss"""
def dice_loss( predicted, truth, epsilon):
    predicted = predicted.detach().numpy()
    truth = truth.detach().numpy()
    numerator = 2. * np.sum(predicted * truth)
    denominator = np.sum(np.square(predicted) + np.square(truth))
    return 1 - np.mean(numerator / (denominator +epsilon))
