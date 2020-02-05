import torch as torch
from tqdm import tqdm
from loss import dice_loss, tversky_loss, compute_loss
from image import save_masks

import logging


def evaluation(model, dataset, device, metrics):
    # Set model modules to eval
    model.eval()
    loss = 0
    with tqdm(desc=f'Validation', unit='img') as progress_bar:
        for image, ground_truth in dataset:
            image = image.to(device)
            ground_truth = ground_truth.to(device)

            with torch.no_grad():
                mask_predicted = model(image)

            progress_bar.set_postfix(**{'loss': loss})
            loss += compute_loss(mask_predicted.type(torch.FloatTensor), ground_truth.type(torch.FloatTensor),
                                 bce_weight=0.5, metrics=metrics).item()

            progress_bar.update()

    loss /= len(dataset)

    return loss
