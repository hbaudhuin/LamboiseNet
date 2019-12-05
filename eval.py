import torch
from tqdm import tqdm
from loss import dice_loss
import logging
def evaluation( model, dataset, device) :
    #Set model modules to eval
    model.eval()
    loss = 0
    with tqdm(desc=f'Validation', unit='img') as progress_bar :
        for image, ground_truth in dataset :

            image.to(device)
            ground_truth.to(device)

            mask_predicted = model(image)

            progress_bar.set_postfix(**{'loss': loss})

            loss += dice_loss(mask_predicted, ground_truth, epsilon = 1e-6)

            progress_bar.update()

    loss /= len(dataset)

    return loss


