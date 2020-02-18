import torch as torch
from tqdm import tqdm
from loss import dice_loss, tversky_loss, compute_loss
from image import save_masks
import torch.nn as nn
import logging


def evaluation(model, dataset, device, metrics):
    # Set model modules to eval
    model.eval()
    loss = 0

    last_masks = [None] * len(dataset)
    last_truths = [None] * len(dataset)

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


            #loss += tversky_loss(ground_truth, mask_predicted[0,0, :,:], beta = 0.85)

            progress_bar.update()

    save_masks(last_masks, last_truths, str(device), max_img=50, shuffle=False, color="red", filename="mask_predicted_test.png")

    loss /= len(dataset)

    return loss
