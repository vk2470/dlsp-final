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


def get_data(percentage_labelled, percentage_unlabelled):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 32

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    labelled_indices = random.sample(range(0, len(trainset)), int(percentage_labelled * len(trainset)))
    unlabelled_indices = [i for i in range(len(trainset)) if i not in labelled_indices]
    unlabelled_indices = random.sample(unlabelled_indices, int(percentage_unlabelled * len(unlabelled_indices)))
    subset = torch.utils.data.Subset(trainset, labelled_indices)
    labelled_trainloader = torch.utils.data.DataLoader(subset, batch_size=1, num_workers=0, shuffle=False)

    subset = torch.utils.data.Subset(trainset, unlabelled_indices)
    unlabelled_trainloader = torch.utils.data.DataLoader(subset, batch_size=1, num_workers=0, shuffle=False)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    return labelled_trainloader, unlabelled_trainloader, testset, testloader


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_uniform_(m.weight)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.layer_0 = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.layer_1 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.layer_3 = nn.Conv2d(32, 64, 4)
        self.encoder = nn.Sequential(  # like the Composition layer you built
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, padding=1),  # , output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, padding=1),  # , output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 3, padding=1),  # , output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class FineTuner(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super(FineTuner, self).__init__()
        self.embedding = pretrained_model
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.finetuner = nn.Sequential(self.fc1,
                                       nn.ReLU(),
                                       self.fc2,
                                       nn.ReLU(),
                                       self.fc3)

    def forward(self, x):
        x = self.embedding(x)
        x = self.finetuner(x)
        return x


def train_model(model, batch_train_loader, optimizer, loss_fn, classification=False):
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


def evaluate_classification(model, batch_test_loader, loss_fn):
    model.eval()
    losses_within_batch = []
    accuracies_within_batch = []
    with torch.zero_grad():
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
    auto_encoder_model = Autoencoder()
    auto_encoder_model.load_state_dict(torch.load(pretrained_model_path))
    auto_encoder_model = auto_encoder_model.to(device)
    for param in auto_encoder_model.parameters():
        param.requires_grad = False

    return auto_encoder_model