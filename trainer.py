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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device is", device)


def get_data(percentage_labelled):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 32

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    labelled_indices = random.sample(range(0, len(trainset)), int(percentage_labelled * len(trainset)))
    unlabelled_indices = [i for i in range(len(trainset)) if i not in labelled_indices]
    subset = torch.utils.data.Subset(trainset, labelled_indices)
    labelled_trainloader = torch.utils.data.DataLoader(subset, batch_size=1, num_workers=0, shuffle=False)

    subset = torch.utils.data.Subset(trainset, unlabelled_indices)
    unlabelled_trainloader = torch.utils.data.DataLoader(subset, batch_size=1, num_workers=0, shuffle=False)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    return labelled_trainloader, unlabelled_trainloader, testset, testloader

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


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


def train_model(train_model, batch_train_loader, optimizer, loss_fn):
    train_model.train()
    losses_within_batch = []
    for i, data in tqdm(enumerate(batch_train_loader), total=len(batch_train_loader), leave=False):
        optimizer.zero_grad()
        input_data = data[0].to(device)
        pred = train_model(input_data)
        tmp_loss = loss_fn(pred, input_data)
        losses_within_batch.append(tmp_loss.item())
        tmp_loss.backward()
        optimizer.step()
    final_loss = np.mean(losses_within_batch)
    return final_loss, train_model


def train(num_epochs, percentage_labelled):
    labelled_trainloader, unlabelled_trainloader, testset, testloader = get_data(percentage_labelled)
    auto_encoder_model = Autoencoder()
    auto_encoder_model = auto_encoder_model.to(device)
    criterion = nn.MSELoss()
    lr = 0.001
    optimizer = optim.Adam(auto_encoder_model.parameters(), lr=lr)
    all_losses = []

    if not os.path.exists('test_images'):
        os.mkdir('test_images')

    for epoch in tqdm(range(num_epochs)):
        loss, auto_encoder_model = train_model(auto_encoder_model, labelled_trainloader, optimizer, criterion)
        print("{} {}".format(epoch, loss))
        all_losses.append(loss)

        subset_indices = random.sample(range(0, len(testset)), 10)
        subset = torch.utils.data.Subset(testset, subset_indices)
        testloader_subset = torch.utils.data.DataLoader(subset, batch_size=1, num_workers=0, shuffle=False)

        for i, data in enumerate(testloader_subset):
            original_data = data[0][0].permute(1, 2, 0)
            original_data = (original_data * 0.5) + 0.5
            plt.imshow(original_data)
            plt.savefig('test_images/{}_{}_original'.format(i, epoch))
            reconstructed_image = auto_encoder_model(data[0].to(device))[0].detach().cpu()
            reconstructed_image = reconstructed_image.permute(1, 2, 0)
            reconstructed_image = (reconstructed_image * 0.5) + 0.5
            plt.imshow(reconstructed_image)
            plt.savefig('test_images/{}_{}_reconstructed'.format(i, epoch))
    return all_losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--percentage_labelled", type=float)
    args = parser.parse_args()
    all_losses = train(args.num_epochs, float(args.percentage_labelled))

