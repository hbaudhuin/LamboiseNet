from sklearn.metrics import precision_recall_fscore_support as prfs

from torch import *
from Models.basicUnet import BasicUnet
from Models.modularUnet import modularUnet
import torch.utils.data
from image import *
import logging
from tqdm import tqdm
import torch.nn as nn
from eval import evaluation
import time
import matplotlib.pyplot as plt
import datetime
from loss import compute_loss, print_metrics


def train_model(model,
                num_epochs,
                batch_size,
                learning_rate,
                device,
                reload,
                save_model):
    logging.info(f'''Starting training : 
                Type : {model.name}
                Epochs: {num_epochs}
                Batch size: {batch_size}
                Learning rate: {learning_rate}
                Device: {device.type}
                Saving model : {save_model}''')

    if reload:
        model.load_state_dict(torch.load('backup_weights/last_backup.pth'))
        # model.load_state_dict(torch.load('Weights/kek.pth'))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    last_masks = [None] * len(train_dataset)
    last_truths = [None] * len(train_dataset)

    accuracies = []

    if reload:
        with open('Loss/last.pth', 'r') as acc_file:
            prev_acc = np.loadtxt('Loss/last.pth')
            for acc in prev_acc:
                accuracies.append(acc)

    for epochs in range(0, num_epochs):
        logging.info(f'Epoch {epochs}')
        torch.autograd.set_detect_anomaly(True)
        accuracy = 0
        #Every epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0
            metrics = dict([("tversky", 0),  ("BCE", 0), ("loss", 0)])

            with tqdm(desc=f'Epoch {epochs}', unit='img') as progress_bar:
                for i, (images, ground_truth) in enumerate(train_dataset):
                    images = images.to(device)
                    last_truths[i] = ground_truth
                    ground_truth = ground_truth.to(device)

                    mask_predicted = model(images)
                    last_masks[i] = mask_predicted

                    # During the training, we backpropagate the error
                    if phase == 'train':
                        loss = compute_loss(mask_predicted.type(torch.FloatTensor),
                                            ground_truth.type(torch.FloatTensor),
                                            bce_weight=0.5, metrics=metrics)
                        epoch_loss += loss.item()
                        progress_bar.set_postfix(**{'loss': loss.item()})

                        # zero the gradient and back propagate
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    # During validation we compute the metrics
                    if phase == 'val':
                        loss = compute_loss(mask_predicted.type(torch.FloatTensor),
                                            ground_truth.type(torch.FloatTensor),
                                            bce_weight=0.5, metrics=metrics)
                        accuracy += loss / len(train_dataset)

                    progress_bar.update(1)

            print_metrics(metrics, len(train_dataset), phase)
            if phase == 'val':
                accuracies.append(accuracy)

    save_masks(last_masks, last_truths, str(device), max_img=20, shuffle=False)

    #Test set evaluation
    metrics = dict([("tversky", 0), ("BCE", 0),("loss", 0)])
    evaluation(model, test_dataset,device,  metrics)
    print_metrics(metrics, len(test_dataset), 'test set')

    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if save_model:
        placeholder_file('Weights/last.pth')
        torch.save(model.state_dict(), 'Weights/last.pth')

        placeholder_file('Weights/' + current_datetime + '.pth')
        torch.save(model.state_dict(), 'Weights/' + current_datetime + '.pth')
        logging.info(f'Model saved')

    placeholder_file(
        'Loss/' + 'learning_' + str(learning_rate) + '_epoch_' + str(num_epochs) + '_time_' + current_datetime + '.pth')
    np.savetxt(
        'Loss/' + 'learning_' + str(learning_rate) + '_epoch_' + str(num_epochs) + '_time_' + current_datetime + '.pth',
        accuracies)
    placeholder_file('Loss/last.pth')
    np.savetxt('Loss/last.pth', accuracies)

    plt.plot([i for i in range(0, len(accuracies))], accuracies)
    plt.xlabel("Epochs")
    plt.ylabel("Cross-Entropy with BCE loss")
    plt.show()
    plt.savefig("Loss.png")
    plt.close("Loss.png")


if __name__ == '__main__':
    t_start = time.time()

    # Hyperparameters
    num_epochs = 2
    num_classes = 2
    batch_size = 1
    learning_rate = 0.01
    n_images = 1
    n_channels = 6

    # setup of log and device
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
    logging.info(f'Using {device}')

    # dataset setup

    # transform into pytorch vector and normalise
    # batch_index= batch(batch_size, n_images)
    train_dataset = load_dataset(IMAGE_NUM[0:2], 2)
    test_dataset = load_dataset(IMAGE_NUM[3:4], 2)
    # train_dataset = load_dataset(IMAGE_NUM)
    # test_dataset = load_dataset(IMAGE_NUM)

    logging.info(f'Batch size: {batch_size}')

    # TODO add check of arguments

    # model creation

    # model = BasicUnet(n_channels= n_channels, n_classes=num_classes)
    model = modularUnet(n_channels=n_channels, n_classes=num_classes, depth=4)
    model.to(device)
    logging.info(f'Network creation:\n')

    # Print the summary of the model
    # summary(model, (6, 650, 650))

try:
    train_model(model=model,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device,
                reload=False,
                save_model=False)


except KeyboardInterrupt:
    # torch.save(model.state_dict(), 'Weights/last.pth')
    logging.info(f'Interrupted by Keyboard')
finally:
    t_end = time.time()
    print("\nDone in " + str(int((t_end - t_start))) + " sec")

# TODO start writing memoire to keep track of source (tqdm https://towardsdatascience.com/progress-bars-in-python-4b44e8a4c482)
# TODO Use GRADCAM !!
