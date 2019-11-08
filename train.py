from sklearn.metrics import precision_recall_fscore_support as prfs

import torch
import torch.utils.data
import torch.optim as optim
import torch.autograd as autograd
import torchvision as tv
from torch.utils.data import *

# Hyperparameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001


DATA_PATH = '.\PycharmProjects\LamboiseNet\Data'
#MODEL_STORE_PATH = '..\PycharmProjects\pytorch_models\\'

#transform into pytorch vector and normalise
trans = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((0.1307,), (0.3081,))])


#Load dataset into pytorch datasets
train_dataset = tv.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
test_dataset = tv.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)