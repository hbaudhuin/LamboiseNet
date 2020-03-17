from sklearn.metrics import precision_recall_fscore_support as prfs

from torch import *
from Models.basicUnet import BasicUnet
from Models.modularUnet import modularUnet
from Models.unetPlusPlus import unetPlusPlus
from Models.lightUnetPlusPlus import lightUnetPlusPlus
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
import torchsummary


def train_model(model,
                num_epochs,
                batch_size,
                learning_rate,
                device,
                train_dataset,
                test_dataset,
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
        #model.load_state_dict(torch.load('backup_weights/last_backup.pth'))
        model.load_state_dict(torch.load('Weights/last.pth'))
        #model.load_state_dict(torch.load('Weights/kek.pth'))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    last_masks = [None] * len(train_dataset)
    last_truths = [None] * len(train_dataset)

    prev_epochs = 0
    losses_train = []
    losses_test = []
    losses_test_19 = []
    losses_test_91 = []

    metrics_idx = []
    auc = []
    f1_score = []

    metrics_idx.append(0)
    auc.append(0.5)
    f1_score.append(0)

    if reload:
        #try:
            prev_loss = np.loadtxt('Loss/last.pth')
            #blank = np.zeros(shape=(prev_loss.shape[0], 4))
            #blank[:, 0] = prev_loss[:, 0]
            #blank[:, 1] = prev_loss[:, 1]
            #prev_loss = blank
            losses_train = list(prev_loss[:, 0])
            losses_test = list(prev_loss[:, 1])
            losses_test_19 = list(prev_loss[:, 2])
            losses_test_91 = list(prev_loss[:, 3])
            prev_epochs = len(losses_train)

            prev_metrics = np.loadtxt('Loss/last_metrics.pth')
            metrics_idx = list(prev_metrics[:, 0])
            auc = list(prev_metrics[:, 1])
            f1_score = list(prev_metrics[:, 2])
        #except:
        #    print("Failed to load previous loss values")

    changed = 5
    for epochs in range(0, num_epochs):

        # new dataset with random augmentation at each epoch
        train_dataset = load_dataset(IMAGE_NUM[0:22], 2, batch_size=batch_size)

        logging.info(f'Epoch {epochs}')
        if len(losses_train) > 100 :
            if np.linalg.norm(losses_train[-1:-4]) < 0.01 and changed < 1:
                changed = 10
                logging.info(f'Learning rate going to {learning_rate/2}')
                learning_rate /= 2
                optimizer.lr = learning_rate
            else:
                changed -= 1
        torch.autograd.set_detect_anomaly(True)

        loss_train = 0
        loss_test = 0
        loss_test_19 = 0
        loss_test_91 = 0

        #Every epoch has a training and validation phase

        # TRAIN
        with tqdm(desc=f'Epoch {epochs}', unit='img') as progress_bar:
            model.train()
            for i, (images, ground_truth) in enumerate(train_dataset):

                images = images[0, ...]
                ground_truth = ground_truth[0, ...]

                images = images.to(device)
                last_truths[i] = ground_truth
                ground_truth = ground_truth.to(device)

                mask_predicted = model(images)

                last_masks[i] = mask_predicted

                bce_weight = torch.Tensor([1, 8]).to(device)
                loss = compute_loss(mask_predicted, ground_truth, bce_weight=bce_weight)

                loss_train += loss.item() / len(train_dataset)
                progress_bar.set_postfix(**{'loss': loss.item()})

                # zero the gradient and back propagate
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress_bar.update(1)

        # TEST

        '''
        with tqdm(desc=f'Epoch {epochs}', unit='img') as progress_bar:

            model.eval()
            for i, (images, ground_truth) in enumerate(test_dataset):

                images = images[0, ...]
                ground_truth = ground_truth[0, ...]

                images = images.to(device)
                ground_truth = ground_truth.to(device)

                with torch.no_grad():
                    mask_predicted = model(images)

                loss = compute_loss(mask_predicted,
                                    ground_truth,
                                    bce_weight=torch.Tensor([0.5, 0.5]).to(device))
                loss19 = compute_loss(mask_predicted,
                                      ground_truth,
                                      bce_weight=torch.Tensor([0.1, 0.9]).to(device))
                loss91 = compute_loss(mask_predicted,
                                      ground_truth,
                                      bce_weight=torch.Tensor([0.9, 0.1]).to(device))
                loss_test += loss / len(test_dataset)
                loss_test_19 += loss19 / len(test_dataset)
                loss_test_91 += loss91 / len(test_dataset)
                progress_bar.set_postfix(**{'loss': loss.item()})

                progress_bar.update(1)
        '''
        test_metrics = evaluation(model, test_dataset, device, save_mask=False, plot_roc=False, print_metric=False)
        loss_test = test_metrics["loss"]

        #print_metrics(metrics, len(train_dataset), phase)
        logging.info(f'Train loss {loss_train}')
        logging.info(f'Test loss  {loss_test}')
        losses_train.append(loss_train)
        losses_test.append(loss_test)
        losses_test_19.append(loss_test_19)
        losses_test_91.append(loss_test_91)

        metrics_idx.append(prev_epochs + epochs)
        auc.append(test_metrics["AUC"])
        f1_score.append(np.max(test_metrics["F1"]))

    save_masks(last_masks, last_truths, str(device), max_img=50, shuffle=False, threshold=test_metrics["best_threshold"])

    #Test set evaluation
    metrics = evaluation(model, test_dataset, device)
    #print_metrics(metrics, len(test_dataset), 'test set')

    #metrics_idx.append(prev_epochs + num_epochs - 1)
    #auc.append(metrics["AUC"])
    #f1_score.append(np.max(metrics["F1"]))

    # save model weights
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if save_model:
        placeholder_file('Weights/last.pth')
        torch.save(model.state_dict(), 'Weights/last.pth')

        placeholder_file('Weights/' + current_datetime + "-" + str(prev_epochs+num_epochs) + '.pth')
        torch.save(model.state_dict(), 'Weights/' + current_datetime + "-" + str(prev_epochs+num_epochs) + '.pth')
        logging.info(f'Model saved')

        # save the losses
        loss_to_save = np.stack([np.asarray(losses_train), np.asarray(losses_test), np.asarray(losses_test_19), np.asarray(losses_test_91)], axis=1)
        placeholder_file(
            'Loss/' + 'learning_' + str(learning_rate) + '_epoch_' + str(num_epochs) + '_time_' + current_datetime + '.pth')
        np.savetxt(
            'Loss/' + 'learning_' + str(learning_rate) + '_epoch_' + str(num_epochs) + '_time_' + current_datetime + '.pth',
            loss_to_save)
        placeholder_file('Loss/last.pth')
        np.savetxt('Loss/last.pth', loss_to_save)

        # save the metrics
        metrics_to_save = np.stack([np.asarray(metrics_idx), np.asarray(auc), np.asarray(f1_score)], axis=1)
        placeholder_file('Loss/last_metrics.pth')
        np.savetxt('Loss/last_metrics.pth', metrics_to_save)

    # plot train and test losses
    plt.plot([i for i in range(0, len(losses_train))], losses_train, label='Train Loss = '+str(round(losses_train[len(losses_train)-1], 3)))
    plt.plot([i for i in range(0, len(losses_test))], losses_test, label='Test Loss = '+str(round(losses_test[len(losses_test)-1].item(), 3)))
    #plt.plot([i for i in range(0, len(losses_test_19))], losses_test_19, label='Test Loss 19 = '+str(round(losses_test_19[len(losses_test_19)-1].item(), 3)))
    #plt.plot([i for i in range(0, len(losses_test_91))], losses_test_91, label='Test Loss 91 = '+str(round(losses_test_91[len(losses_test_91)-1].item(), 3)))
    plt.plot(metrics_idx, [1-auc_ for auc_ in auc], label='1 - AUC (AUC = '+ str(round(float(auc[len(auc)-1]), 3)) +')')
    plt.plot(metrics_idx, [1-f1 for f1 in f1_score], label='1 - F1 (F1 = '+ str(round(float(f1_score[len(f1_score)-1]), 3)) +')')
    plt.legend()
    plt.ylim(bottom=0, top=1)
    plt.xlabel("Epochs")
    plt.ylabel("Metric")
    plt.savefig("Loss.png")
    plt.show()
    plt.close("Loss.png")



if __name__ == '__main__':
    t_start = time.time()

    # Hyperparameters
    num_epochs = 10
    num_classes = 2
    batch_size = 1
    learning_rate = 0.0002
    n_images = 1
    n_channels = 6

    # setup of log and device
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
    logging.info(f'Using {device}')

    # dataset setup
    logging.info(f'Generating dataset ...')
    logging.info(f'Batch size: {batch_size}')

    # transform into pytorch vector and normalise
    # batch_index= batch(batch_size, n_images)
    train_dataset = load_dataset(IMAGE_NUM[0:22], 2, batch_size)
    test_dataset = load_dataset(IMAGE_NUM[22:32], 0)
    # train_dataset = load_dataset(IMAGE_NUM)
    # test_dataset = load_dataset(IMAGE_NUM)

    logging.info(f'Dataset generated')

    # TODO add check of arguments

    # model creation

    #model = BasicUnet(n_channels= n_channels, n_classes=num_classes)
    #model = modularUnet(n_channels=n_channels, n_classes=num_classes, depth=4)
    #model = unetPlusPlus(n_channels=n_channels, n_classes=num_classes)
    model = lightUnetPlusPlus(n_channels=n_channels, n_classes=num_classes)
    model.to(device)
    logging.info(f'Network creation:\n')

    # Print the summary of the model
    #torchsummary.summary(model, (6, 650, 650))

try:
    train_model(model=model,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                reload=True,
                save_model=True)


except KeyboardInterrupt:
    torch.save(model.state_dict(), 'Weights/last.pth')
    logging.info(f'Interrupted by Keyboard')
finally:
    t_end = time.time()
    print("\nDone in " + str(int((t_end - t_start))) + " sec")

# TODO start writing memoire to keep track of source (tqdm https://towardsdatascience.com/progress-bars-in-python-4b44e8a4c482)
# TODO Use GRADCAM !!
