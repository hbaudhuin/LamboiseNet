import torch as torch
from tqdm import tqdm
from loss import dice_loss, tversky_loss
from image import save_masks


import logging
def evaluation( model, dataset, device, criterion) :
    #Set model modules to eval
    model.eval()
    loss = 0
    with tqdm(desc=f'Validation', unit='img') as progress_bar :
        for image, ground_truth in dataset :

            image = image.to(device)
            ground_truth = ground_truth.to(device)

            with torch.no_grad():
                mask_predicted = model(image)




            progress_bar.set_postfix(**{'loss': loss})
            loss +=criterion(mask_predicted, ground_truth).item()

            #loss += tversky_loss(ground_truth, mask_predicted[0,0, :,:], beta = 0.85)

            progress_bar.update()

    loss /= len(dataset)

    return loss


