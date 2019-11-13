import torch
import tqdm
from loss import dice_loss

def evaluation( model, dataset, device) :
    #Set model modules to eval
    model.eval()
    loss =  0
    with tqdm(desc=f'Validation', unit='img') as progress_bar :
        for i, data in enumerate(dataset) :
            image = torch.from_numpy(data[0])
            ground_truth = torch.from_numpy(data[1])

            image.to(device)
            ground_truth.to(device)

            mask_predicted = model(image)

            progress_bar.set_postfix(**{'loss': loss})
            loss +=dice_loss(mask_predicted, ground_truth, epsilon = 1e-6)

            progress_bar.update()

    #TODO à 100% c'est pas ça mais je dois avoir les images pour check
    loss /= len(dataset)

    return loss


