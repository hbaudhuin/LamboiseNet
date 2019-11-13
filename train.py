from sklearn.metrics import precision_recall_fscore_support as prfs

from torch import *
from Models import basicUnet
import torch.utils.data
import torch.optim as optim
import torch.autograd as autograd
import torchvision as tv
from torch.utils.data import *
from image import *
import logging
import tqdm
from eval import evaluation




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

    #TODO check if it's the best optimizer for our case

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epochs in range(num_epochs) :
        #state intent of training to the model
        model.train()

        epoch_loss= 0

        with tqdm(desc = f'Epoch {epochs}', unit = 'img') as progress_bar :
            #TODO add batches

            for i, data in enumerate(train_dataset):
                #TODO check input format
                images = torch.from_numpy()
                ground_truth = torch.from_numpy()

                images.to(device)
                ground_truth.to(device)

                mask_predicted = model(images)

                loss  = criterion(mask_predicted, ground_truth)
                epoch_loss +=loss.item()
                progress_bar.set_postfix(**{'loss': loss.item()})

                #zero the gradient and back propagate
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #TODO update necessary for progress_bar ?

        #TODO add eval methods
        #TODO what is the evaluation metric

        logging.info(f'Loss at  {epochs} : {epoch_loss}')

        evaluation(model, test_dataset, device)







if __name__ == '__main__':

    # Hyperparameters
    num_epochs = 5
    num_classes = 10
    batch_size = 100
    learning_rate = 0.001

    #setup of log and device
    logging.basicConfig(level = logging.INFO, format = '%(level)s: %(message)s')
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
    logging.info(f'Using {device}')

    #dataset setup


    #transform into pytorch vector and normalise
    img1 = open_image("/DATA/Paris_1180/before.png")
    img2 = open_image("/DATA/Paris_1180/after.png")
    msk = open_image("/DATA/Paris_1180/mask.png")
    arrs = [None, None, None]
    for i in range(3):
        arr = np.copy(img1[..., i])


    image=  cat(img1, img2)
    train_dataset = open_image("/DATA/Paris_1180/before")
    test_dataset = get_singleExample(DATA_PATH)



    #TODO add check of arguments

    #model creation

    model = basicUnet(n_channelse = 3, n_classes = 1)
    logging.info(f'Netword creation:\n', f'\t{model.n_channels} input channels\n' f'\t{model.n_classes} output channels\n')
    model.to(device)


try :
    train_model(model = model,
                epochs = num_epochs,
                batch_size = batch_size,
                learning_rate= learning_rate,
                device=device)
except KeyboardInterrupt :
    torch.save(model.state_dict(), 'INTERUPTED')
    logging.info(f'Interupted by Keyboard')




#TODO start writing memoire to keep track of source (tqdm https://towardsdatascience.com/progress-bars-in-python-4b44e8a4c482)