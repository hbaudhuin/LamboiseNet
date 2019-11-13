import torch
import numpy as np

#TODO add other loss compution (jaccard, Tversky)


#TODO variate epsilon
"""SOFT dice loss
This functiun comes from https://www.jeremyjordan.me/semantic-segmentation/"""
def dice_loss( predicted, truth, epsilon):
    axes = tuple(range(1, len(predicted.shape)-1))
    numerator = 2. *np.sum(predicted * truth, axes)
    denominator = np.sum(np.square(predicted) + np.square(truth), axes)
    return 1 - np.mean(numerator / (denominator +epsilon))
