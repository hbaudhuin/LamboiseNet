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
from torchsummary import summary
import time
import matplotlib.pyplot as plt
import datetime

def train_model(model,
                num_epochs,
                batch_size,
                learning_rate,
                device, reload):
    logging.info(f'''Strarting training : 
                Type : {model.name}
                Epochs: {num_epochs}
                Batch size: {batch_size}
                Learning rate: {learning_rate}
                Device: {device.type}''')

    # TODO check if it's the best optimizer for our case
    if reload :
        model.load_state_dict(torch.load('Weights/last.pth'))
        #model.load_state_dict(torch.load('Weights/kek.pth'))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    transform = torchvision.transforms.Normalize(mean=0, std=1)
    last_masks = [None] * len(train_dataset)
    last_truths = [None] * len(train_dataset)

    accuracies = []
    if reload:
        with open('Loss/last.pth', 'r') as acc_file:
            prev_acc = np.loadtxt('Loss/last.pth')
            for acc in prev_acc:
                accuracies.append(acc)

    for epochs in range(num_epochs):
        # state intent of training to the model
        model.train()

        epoch_loss = 0
        torch.autograd.set_detect_anomaly(True)
        with tqdm(desc=f'Epoch {epochs}', unit='img') as progress_bar:

            for i, (images, ground_truth) in enumerate(train_dataset):


                images = images.to(device)
                last_truths[i] = ground_truth
                ground_truth = ground_truth.to(device)

                mask_predicted = model(images)
                last_masks[i] = mask_predicted

                loss = criterion(mask_predicted, ground_truth)
                epoch_loss += loss.item()
                progress_bar.set_postfix(**{'loss': loss.item()})

                # zero the gradient and back propagate
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress_bar.update(1)

        # TODO add eval methods
        # TODO what is the evaluation metric

        logging.info(f'Loss at  {epochs} : {epoch_loss/len(train_dataset)}')

        score = evaluation(model, test_dataset, device, criterion)

        accuracies.append(score)


    #save_masks(last_masks, last_truths, max_img=100, shuffle=False)

    torch.save(model.state_dict(), 'Weights/last.pth')
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    placeholder_file('Weights/' + current_datetime + '.pth')
    torch.save(model.state_dict(), 'Weights/' + current_datetime + '.pth')

    logging.info(f'Model saved')
    score = evaluation(model, test_dataset, device, criterion)

    logging.info(f'Validation score (soft dice method): {score}')

    placeholder_file('Loss/' + 'learning_' +str(learning_rate) + '_epoch_' + str(num_epochs) + '_time_' + current_datetime + '.pth')
    np.savetxt('Loss/' + 'learning_' +str(learning_rate) +'_epoch_'+ str(num_epochs)+ '_time_'+ current_datetime+'.pth', accuracies)
    placeholder_file('Loss/last.pth')
    np.savetxt('Loss/last.pth', accuracies)

    plt.plot( [i for i in range(0, len(accuracies))], (accuracies))
    plt.xlabel("Epochs")
    plt.ylabel("Cross-Entropy loss")
    plt.show()
    plt.savefig("Loss.png")


if __name__ == '__main__':
    t_start = time.time()

    # Hyperparameters
    num_epochs = 10
    num_classes = 2
    batch_size = 1
    learning_rate = 0.0001
    n_images = 1
    n_channels = 6

    # setup of log and device
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
    logging.info(f'Using {device}')

    # dataset setup

    # transform into pytorch vector and normalise
    #batch_index= batch(batch_size, n_images)
    train_dataset = load_dataset(IMAGE_NUM)
    test_dataset = load_dataset(IMAGE_NUM)


    logging.info(f'Batch size: {batch_size}')

    # TODO add check of arguments

    # model creation

    model = BasicUnet(n_channels= n_channels, n_classes=num_classes)
    logging.info(f'Network creation:\n' )
      #               f'\t6 input channels\n', f'\t2 output channels\n')
    model.to(device)

    #Print the summary of the model
    #summary(model, (6, 650, 650))

try:
    train_model(model=model,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device,
                reload = True)


except KeyboardInterrupt:
    torch.save(model.state_dict(), 'Weights/last.pth')
    logging.info(f'Interrupted by Keyboard')
finally:
    t_end = time.time()
    print("\nDone in " + str(int((t_end - t_start))) + " sec")

# TODO start writing memoire to keep track of source (tqdm https://towardsdatascience.com/progress-bars-in-python-4b44e8a4c482)
# TODO Use GRADCAM !!
