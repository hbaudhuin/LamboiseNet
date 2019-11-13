from sklearn.metrics import precision_recall_fscore_support as prfs

from torch import *
from Models.basicUnet import BasicUnet
import torch.utils.data
import torch.optim as optim
import torch.autograd as autograd
import torchvision as tv
from torch.utils.data import *
from image import *
import logging
from tqdm import tqdm
from torch.nn import *
import torch.nn as nn
from eval import evaluation
from helpers.batching import batch
import torchvision
import os


def train_model(model,
                num_epochs,
                batch_size,
                learning_rate,
                device):
    logging.info(f'''Strarting training : 
                Type : {model.name}
                Epochs: {num_epochs}
                Batch size: {batch_size}
                Learning rate: {learning_rate}
                Device: {device.type}''')

    # TODO check if it's the best optimizer for our case

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    transform = torchvision.transforms.Normalize(mean=0, std=1)
    last_mask = None
    for epochs in range(num_epochs):
        # state intent of training to the model
        model.train()

        epoch_loss = 0
        torch.autograd.set_detect_anomaly(True)
        with tqdm(desc=f'Epoch {epochs}', unit='img') as progress_bar:

            for images, ground_truth  in train_dataset:
                # TODO check input format

                images.to(device)
                ground_truth.to(device)

                mask_predicted = model(images)
                last_mask = mask_predicted

                loss = criterion(mask_predicted, ground_truth)
                epoch_loss += loss.item()
                progress_bar.set_postfix(**{'loss': loss.item()})

                # zero the gradient and back propagate
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress_bar.update(1)

                # TODO update necessary for progress_bar ?

        # TODO add eval methods
        # TODO what is the evaluation metric

        logging.info(f'Loss at  {epochs} : {epoch_loss}')
    save_mask_predicted(last_mask)

        #torch.save(model.state_dict(), 'Weights/h.pth')

    #score = evaluation(model, test_dataset, device)

    # logging.info(f'Validation score (soft dice method): {score}')


if __name__ == '__main__':
    # Hyperparameters
    num_epochs = 10
    num_classes = 2
    batch_size = 1
    learning_rate = 0.001
    n_images = 1
    n_channels =6

    # setup of log and device
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
    logging.info(f'  Using  {device}')

    # dataset setup

    # transform into pytorch vector and normalise
    #batch_index= batch(batch_size, n_images)
    train_dataset = load_dataset([1180, 1180])
    test_dataset = load_dataset([1180])


    logging.info(f'Batch size: {batch_size}')

    # TODO add check of arguments

    # model creation

    model = BasicUnet(n_channels= n_channels, n_classes=num_classes)
    logging.info(f'Network creation:\n' )
      #               f'\t6 input channels\n', f'\t2 output channels\n')
    model.to(device)

try:
    train_model(model=model,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device)
except KeyboardInterrupt:
    torch.save(model.state_dict(), 'Weights/kek.pth')
    logging.info(f'Interrupted by Keyboard')

# TODO start writing memoire to keep track of source (tqdm https://towardsdatascience.com/progress-bars-in-python-4b44e8a4c482)
