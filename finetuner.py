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
from utils import *


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def finetuner_wrapper(finetuner_model, num_epochs, labelled_trainloader, testloader, optimizer, criterion, folder_name):
    tic = time.time()
    all_train_losses = []
    all_test_losses = []
    all_train_accuracies = []
    all_test_accuracies = []
    for epoch in tqdm(range(num_epochs), leave=False):
        loss, train_accuracy, finetuner_model = train_model(finetuner_model, labelled_trainloader, optimizer,
                                                                  criterion, classification=True)
        tqdm.write("epoch: {} train loss: {} time elapsed: {}".format(epoch, loss, time.time() - tic))
        all_train_losses.append(loss)
        all_train_accuracies.append(train_accuracy)
        json.dump(all_train_losses, open("{}/epoch_{}_loss.json".format(folder_name, epoch), 'w'))
        torch.save(finetuner_model.state_dict(), "{}/epoch_{}.pt".format(folder_name, epoch))
        test_loss, test_accuracy = evaluate_classification(finetuner_model, testloader, criterion)
        all_test_losses.append(test_loss)
        all_test_accuracies.append(test_accuracy)
    return all_train_losses, all_train_accuracies, all_test_losses, all_test_accuracies


def finetune(auto_encoder_model, finetuner_num_epochs, labelled_trainloader, testloader):
    lr = 0.001
    finetuner = FineTuner(auto_encoder_model, len(classes))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(finetuner.parameters(), lr=lr)

    folder_name = '{}_{}_test_images'.format(percentage_labelled, percentage_unlabelled)

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    folder_name = "{}/finetuner".format(folder_name)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    all_train_losses, all_train_accuracies, all_test_losses, all_test_accuracies = finetuner_wrapper(finetuner,
                                                                                               finetuner_num_epochs,
                                                                                               labelled_trainloader,
                                                                                               testloader, optimizer,
                                                                                               criterion, folder_name)
    return all_train_losses, all_train_accuracies, all_test_losses, all_test_accuracies


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--finetuner_num_epochs", type=int)
    parser.add_argument("--percentage_labelled", type=float)
    parser.add_argument("--percentage_unlabelled", type=float)
    args = parser.parse_args()
    percentage_labelled = float(args.percentage_labelled)
    percentage_unlabelled = float(args.percentage_unlabelled)

    labelled_trainloader, unlabelled_trainloader, testset, testloader = get_data(percentage_labelled,
                                                                                 percentage_unlabelled)

    pretrained_model_path = args.pretrained_model_path
    finetuner_num_epochs = args.finetuner_num_epochs
    auto_encoder_model = load_pretrained_model(pretrained_model_path)
    all_train_losses, all_train_accuracies, all_test_losses, all_test_accuracies = finetune(auto_encoder_model,
                                                                                            finetuner_num_epochs,
                                                                                            labelled_trainloader,
                                                                                            testloader)
