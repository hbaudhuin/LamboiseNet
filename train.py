from torch import *
from Models.basicUnet import BasicUnet
from Models.modularUnet import modularUnet
from Models.unetPlusPlus import unetPlusPlus
from Models.lightUnetPlusPlus import lightUnetPlusPlus
import torch.utils.data
from image import *
import logging
from tqdm import tqdm
from eval import evaluation
import time
import matplotlib.pyplot as plt
import datetime
from loss import compute_loss, print_metrics
import argparse


def train_model(model,
                num_epochs,
                batch_size,
                learning_rate,
                device,
                n_augmentation,
                train_dataset,
                test_dataset,
                reload,
                save_model):

    logging.info(f'''Starting training : 
                Type : {model.name}
                Epochs: {num_epochs}
                Batch size: {batch_size}
                Data Augmentation: {n_augmentation}
                Learning rate: {learning_rate}
                Device: {device.type}
                Reloading model : {reload}
                Saving model : {save_model}''')

    # Variables initialization

    if reload:
        model.load_state_dict(torch.load('Weights/last.pth'))
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

    # Reloading previous runs
    if reload:
        try:
            prev_loss = np.loadtxt('Loss/last.pth')
            losses_train = list(prev_loss[:, 0])
            losses_test = list(prev_loss[:, 1])
            losses_test_19 = list(prev_loss[:, 2])
            losses_test_91 = list(prev_loss[:, 3])
            prev_epochs = len(losses_train)

            prev_metrics = np.loadtxt('Loss/last_metrics.pth')
            metrics_idx = list(prev_metrics[:, 0])
            auc = list(prev_metrics[:, 1])
            f1_score = list(prev_metrics[:, 2])
        except:
            print("Failed to load previous loss values")

    changed = 10

    # EPOCH MAIN LOOP
    for epochs in range(0, num_epochs):

        # New dataset with random augmentation at each epoch
        train_dataset = load_dataset(IMAGE_NUM[0:22], n_augmentation, batch_size=batch_size)

        # Adaptive learning rate
        logging.info(f'Epoch {epochs}')
        if len(losses_train) > 100:
            if np.linalg.norm(losses_train[-1:-4]) < 0.01 and changed < 1:
                changed = 10
                learning_rate /= 2
                logging.info(f'Learning rate going to {learning_rate}')
                optimizer.lr = learning_rate
            else:
                changed -= 1
        torch.autograd.set_detect_anomaly(True)

        loss_train = 0
        loss_test = 0
        loss_test_19 = 0
        loss_test_91 = 0

        # Every epoch has a training and validation phase

        # TRAIN
        with tqdm(desc=f'Epoch {epochs}', unit='img') as progress_bar:
            model.train()
            for i, (images, ground_truth) in enumerate(train_dataset):

                # Get the correct data from the dataloader
                images = images[0, ...]
                ground_truth = ground_truth[0, ...]

                # Upload the images to the device
                images = images.to(device)
                last_truths[i] = ground_truth # Keep track to save the masks as images
                ground_truth = ground_truth.to(device)

                # Forward propagation
                mask_predicted = model(images)
                last_masks[i] = mask_predicted # Keep track to save the masks as images

                # Compute loss
                bce_weight = torch.Tensor([1, 8]).to(device)
                loss = compute_loss(mask_predicted, ground_truth, bce_weight=bce_weight)

                loss_train += loss.item() / len(train_dataset)
                progress_bar.set_postfix(**{'loss': loss.item()})

                # Zero the gradient and back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress_bar.update(1)

        # TEST
        test_metrics = evaluation(model, test_dataset, device, save_mask=False, plot_roc=False, print_metric=False)
        loss_test = test_metrics["loss"]

        # Metrics bookkeeping
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

    # END OF EPOCH MAIN LOOP

    # Save the predicted masks in an image
    save_masks(last_masks, last_truths, str(device), max_img=50, shuffle=False, threshold=test_metrics["best_threshold"])

    # Save model weights and metrics
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if save_model:
        placeholder_file('Weights/last.pth')
        torch.save(model.state_dict(), 'Weights/last.pth')

        placeholder_file('Weights/' + current_datetime + "-" + str(prev_epochs+num_epochs) + '.pth')
        torch.save(model.state_dict(), 'Weights/' + current_datetime + "-" + str(prev_epochs+num_epochs) + '.pth')
        logging.info(f'Model saved')

        # Save losses
        loss_to_save = np.stack([np.asarray(losses_train), np.asarray(losses_test), np.asarray(losses_test_19), np.asarray(losses_test_91)], axis=1)
        placeholder_file(
            'Loss/' + 'learning_' + str(learning_rate) + '_epoch_' + str(num_epochs) + '_time_' + current_datetime + '.pth')
        np.savetxt(
            'Loss/' + 'learning_' + str(learning_rate) + '_epoch_' + str(num_epochs) + '_time_' + current_datetime + '.pth',
            loss_to_save)
        placeholder_file('Loss/last.pth')
        np.savetxt('Loss/last.pth', loss_to_save)

        # Save other metrics
        metrics_to_save = np.stack([np.asarray(metrics_idx), np.asarray(auc), np.asarray(f1_score)], axis=1)
        placeholder_file('Loss/last_metrics.pth')
        np.savetxt('Loss/last_metrics.pth', metrics_to_save)

    # Plot train and test losses and metrics
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

    # FINI !



if __name__ == '__main__':
    t_start = time.time()

    # Hyperparameters
    # Change them here
    num_epochs = 1
    learning_rate = 0.001
    batch_size = 1
    n_augmentation = 2
    num_classes = 2
    n_channels = 6

    reload_model = False
    save_model = False

    # Arg parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help="number of epochs the model will run (1 by default)", type=int)
    parser.add_argument("--learning_rate", help="starting learning rate (0.001 by default)", type=float)
    parser.add_argument("--n_data_augm", help="number of data augmentation instances to generate per original instance (2 by default)", type=int)
    parser.add_argument("-reload", help="reload the model weights and metrics from the last run (disabled by default)", action="store_true")
    parser.add_argument("-save", help="save the weights and metrics of the model when it has finished running (disabled by default)", action="store_true")
    args = parser.parse_args()
    if args.epochs is not None:
        num_epochs = args.epochs
    if args.learning_rate is not None:
        learning_rate = args.learning_rate
    if args.n_data_augm is not None:
        n_augmentation = args.n_data_augm
    if args.reload:
        reload_model = True
    if args.save:
        save_model = True

    # Setup of log and device
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
    logging.info(f'Using {device}')

    # Dataset setup !!! It is hardcoded in other part of the code !!!
    # If you want to change this, you have to modify load_dataset and
    # every location where load_dataset is called !!!
    logging.info(f'Generating dataset ...')
    logging.info(f'Batch size: {batch_size}')

    train_dataset = load_dataset(IMAGE_NUM[0:22], n_augmentation, batch_size)
    test_dataset = load_dataset(IMAGE_NUM[22:32], 0)

    logging.info(f'Dataset generated')

    # Network creation, uncomment the one you want to use

    #model = BasicUnet(n_channels= n_channels, n_classes=num_classes)
    #model = modularUnet(n_channels=n_channels, n_classes=num_classes, depth=2)
    #model = unetPlusPlus(n_channels=n_channels, n_classes=num_classes)
    model = lightUnetPlusPlus(n_channels=n_channels, n_classes=num_classes)
    model.to(device)
    logging.info(f'Network creation:\n')



try:
    train_model(model=model,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device,
                n_augmentation=n_augmentation,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                reload=reload_model,
                save_model=save_model)


except KeyboardInterrupt:
    torch.save(model.state_dict(), 'Weights/last.pth')
    logging.info(f'Interrupted by Keyboard')
finally:
    t_end = time.time()
    print("\nDone in " + str(int((t_end - t_start))) + " sec")
