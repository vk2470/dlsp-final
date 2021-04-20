import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch import optim
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import random
import os
import argparse
import time
import json
from sklearn.metrics import f1_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_data(percentage_labelled, percentage_unlabelled, batch_size=32):
    """
    Create the dataloader according to different percentages for labelled and unlabelled data.
    First set aside the labelled data according to percentage_labelled %
    Then choose percentage_unlabelled % of the data from the unlabelled data.

    :param percentage_labelled: float representing how much of the data to be used as labelled data
    :param percentage_unlabelled: float representing how much of the remaining data (after setting aside labelled data
    to be set as unlabelled data)
    :param batch_size: int representing batch size
    :return: labelled_trainloader, unlabelled_trainloader, testset and testloader
    """

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         transforms.RandomApply([transforms.GaussianBlur(5)], p=0.5),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomVerticalFlip(p=0.5)
         ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    all_labelled_datasets = []
    all_unlabelled_datasets = []
    for each_target in set(trainset.targets):
        tmp_trainset_indices = [i for i, each in enumerate(trainset.targets) if each == each_target]
        # tmp_trainset_indices = (trainset.targets == each_target)
        # tmp_trainset_indices = [idx for idx, val in tmp_trainset_indices if idx == True]
        labelled_indices = random.sample(range(0, len(tmp_trainset_indices)),
                                         int(percentage_labelled * len(tmp_trainset_indices)))
        unlabelled_indices = [i for i in range(len(tmp_trainset_indices)) if i not in labelled_indices]
        subset = torch.utils.data.Subset(trainset, labelled_indices)
        all_labelled_datasets.extend(subset)

        if percentage_unlabelled > 0:
            unlabelled_indices = random.sample(unlabelled_indices, int(percentage_unlabelled * len(unlabelled_indices)))
            subset = torch.utils.data.Subset(trainset, unlabelled_indices)
            all_unlabelled_datasets.extend(subset)


    labelled_trainloader = torch.utils.data.DataLoader(all_labelled_datasets, batch_size=batch_size, num_workers=0,
                                                       shuffle=False)
    if percentage_unlabelled > 0:
        unlabelled_trainloader = torch.utils.data.DataLoader(all_unlabelled_datasets, batch_size=batch_size,
                                                             num_workers=0, shuffle=False)
    else:
        unlabelled_trainloader = None

    # subset = torch.utils.data.Subset(trainset, unlabelled_indices)
    # unlabelled_trainloader = torch.utils.data.DataLoader(subset, batch_size=1, num_workers=0, shuffle=False)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    return labelled_trainloader, unlabelled_trainloader, testset, testloader


def weights_init(m):
    """
    Deprecated: Initialize the weights according to kaiming (He) uniform distribution for all conv layers.
    :param m: model
    :return: None
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_uniform_(m.weight)


class Autoencoder(nn.Module):
    """
    AutoEncoder (pretrainer) model
    """
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder_layer_0 = nn.Conv2d(3, 16, 3, padding=1, stride=2)
        self.encoder_layer_1 = nn.Conv2d(16, 32, 3, padding=1, stride=2)
        self.relu = nn.ReLU()
        self.encoder_layer_2 = nn.Conv2d(32, 64, 5)
        self.encoder = nn.Sequential(  # like the Composition layer you built
            self.encoder_layer_0,
            self.relu,
            self.encoder_layer_1,
            self.relu,
            self.encoder_layer_2
        )

        self.decoder_layer_0 = nn.ConvTranspose2d(64, 32, 5)
        self.decoder_layer_1 = nn.ConvTranspose2d(32, 16, 3, padding=1, output_padding=1, stride=2)
        self.decoder_layer_2 = nn.ConvTranspose2d(16, 3, 3, padding=1, output_padding=1, stride=2)

        self.tanh = nn.Tanh()

        self.decoder = nn.Sequential(
            self.decoder_layer_0,
            self.relu,
            self.decoder_layer_1,
            self.relu,
            self.decoder_layer_2,
            self.tanh
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class FineTuner(nn.Module):
    """
    Finetuner model
    """
    def __init__(self, pretrained_model, num_classes):
        super(FineTuner, self).__init__()
        self.embedding = pretrained_model
        self.conv1 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 4 * 4, num_classes)
        self.finetuner = nn.Sequential(self.conv1, nn.ReLU(), nn.Dropout(p=0.4), self.conv2, nn.ReLU(), nn.Dropout(0.4))
        self.fc_layers = nn.Sequential(self.fc1)

    def forward(self, x):
        x = self.embedding.encoder(x) #64*4*4
        x = self.finetuner(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc_layers(x)
        return x


class BaseLine(nn.Module):
    """
    BaseLine model
    """
    def __init__(self, num_classes):
        super(BaseLine, self).__init__()

        self.encoder_layer_0 = nn.Conv2d(3, 16, 3, padding=1, stride=2)
        self.encoder_layer_1 = nn.Conv2d(16, 32, 3, padding=1, stride=2)
        self.relu = nn.ReLU()
        self.encoder_layer_2 = nn.Conv2d(32, 64, 5)
        self.encoder = nn.Sequential(  # like the Composition layer you built
            self.encoder_layer_0,
            self.relu,
            self.encoder_layer_1,
            self.relu,
            self.encoder_layer_2
        )

        self.conv1 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 4 * 4, num_classes)
        self.finetuner = nn.Sequential(self.conv1, nn.ReLU(), nn.Dropout(p=0.4), self.conv2, nn.ReLU(), nn.Dropout(0.4))
        self.fc_layers = nn.Sequential(self.fc1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.finetuner(x)
        # print(x.shape)
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc_layers(x)
        return x


def train_model(model, batch_train_loader, optimizer, loss_fn, classification=False):
    """
    Generic function to train a given model, given an optimizer and a loss function. Perform gradient descent and
    update weights
    :param model: nn.Module instance - model to be trained
    :param batch_train_loader: Training data nn.data.Dataloader instance
    :param optimizer: torch.optim instance of the optimizer
    :param loss_fn: loss function
    :param classification: Boolean representing whether we are doing a regression loss or a classification loss
    :return: final_loss: mean of losses across batches, final_accuracy: mean of accuracy across batches (for classification)
    model: updated model.
    """
    model.train()
    losses_within_batch = []
    accuracies_within_batch = []
    for i, data in tqdm(enumerate(batch_train_loader), total=len(batch_train_loader), leave=False):
        optimizer.zero_grad()
        input_data = data[0].to(device)
        pred = model(input_data)
        if classification:
            labels = data[1].to(device)
            tmp_loss = loss_fn(pred, labels)
            pred = np.argmax(pred.detach().cpu(), 1)
            f1 = f1_score(labels.cpu(), pred, average='macro')
            accuracies_within_batch.append(f1)
        else:
            tmp_loss = loss_fn(pred, input_data)
        losses_within_batch.append(tmp_loss.item())
        tmp_loss.backward()
        optimizer.step()
    final_loss = np.mean(losses_within_batch)
    final_accuracy = np.mean(accuracies_within_batch)
    return final_loss, final_accuracy, model


def evaluate_autoencoder(model, batch_test_loader, loss_fn):
    """
    Generic function to evaluate a given regression model with test set.
    :param model:
    :param batch_test_loader:
    :param loss_fn: loss function
    :return: average loss across all batches
    """
    model.eval()
    losses_within_batch = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(batch_test_loader), total=len(batch_test_loader), leave=False):
            input_data = data[0].to(device)
            pred = model(input_data)
            tmp_loss = loss_fn(pred, input_data)
            losses_within_batch.append(tmp_loss.item())
        final_loss = np.mean(losses_within_batch)
    return final_loss


def evaluate_classification(model, batch_test_loader, loss_fn):
    """
    Generic function to evaluate a given regression model with test set.
    :param model:
    :param batch_test_loader:
    :param loss_fn: loss function
    :return: average loss across all batches
    """
    model.eval()
    losses_within_batch = []
    accuracies_within_batch = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(batch_test_loader), total=len(batch_test_loader), leave=False):
            input_data = data[0].to(device)
            pred = model(input_data)
            labels = data[1].to(device)
            tmp_loss = loss_fn(pred, labels)
            losses_within_batch.append(tmp_loss.item())
            pred = np.argmax(pred.detach().cpu(), 1)
            f1 = f1_score(labels.cpu(), pred, average='macro')
            accuracies_within_batch.append(f1)
    final_loss = np.mean(losses_within_batch)
    final_accuracy = np.mean(accuracies_within_batch)
    return final_loss, final_accuracy

def load_pretrained_model(pretrained_model_path):
    """
    Utility to load an existing model from a path
    :param pretrained_model_path: path of existing model
    :return: model instance
    """
    auto_encoder_model = Autoencoder()
    auto_encoder_model.load_state_dict(torch.load(pretrained_model_path))
    auto_encoder_model = auto_encoder_model.to(device)
    for param in auto_encoder_model.parameters():
        param.requires_grad = False


    return auto_encoder_model