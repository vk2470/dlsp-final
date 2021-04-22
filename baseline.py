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


def baseline_wrapper(baseline_model, num_epochs, labelled_trainloader, testloader, optimizer, criterion, folder_name):
    """
    Wrapper function that calls train_model and evaluate_classification from utils.py
    :param baseline_model: Model instance of the baseline model
    :param num_epochs: Total number of epochs for training
    :param labelled_trainloader: Dataloader object for training
    :param testloader: Dataloader object for testing
    :param optimizer: Optimizer object
    :param criterion: Loss function object
    :param folder_name: Folder where loss and accuracy values are stored
    :return: train losses list, train accuracy list, test loss list and test accuracy list
    """
    tic = time.time()
    all_train_losses = []
    all_test_losses = []
    all_train_accuracies = []
    all_test_accuracies = []
    prev_loss = np.inf
    stored_accuracy = False
    for epoch in tqdm(range(num_epochs), leave=False):
        loss, train_accuracy, baseline_model = train_model(baseline_model, labelled_trainloader, optimizer,
                                                            criterion, classification=True)
        all_train_losses.append(loss)
        all_train_accuracies.append(train_accuracy)
        json.dump(all_train_losses, open("{}/epoch_{}_loss.json".format(folder_name, epoch), 'w'))
        json.dump(all_train_accuracies, open("{}/epoch_{}_accuracy.json".format(folder_name, epoch), 'w'))
        test_loss, test_accuracy = evaluate_classification(baseline_model, testloader, criterion)
        tqdm.write(
            "epoch: {} train loss: train accuracy: {} {} test loss: {} test accuracy: {} time elapsed: {}".format(epoch,
                                                                                                                  loss,
                                                                                                                  train_accuracy,
                                                                                                                  test_loss,
                                                                                                                  test_accuracy,
                                                                                                                  time.time() - tic))
        all_test_losses.append(test_loss)
        all_test_accuracies.append(test_accuracy)
        json.dump(all_test_losses, open("{}/epoch_{}_test_loss.json".format(folder_name, epoch), 'w'))
        json.dump(all_test_accuracies, open("{}/epoch_{}_test_accuracy.json".format(folder_name, epoch), 'w'))
        if loss < prev_loss:
            prev_loss = loss
            torch.save(baseline_model.state_dict(), "{}/baseline.pt".format(folder_name))
        if test_accuracy > 0.40 and not stored_accuracy:
            stored_accuracy = True
            json.dump([time.time() - tic, epoch], open("{}/time_to_accuracy.json".format(folder_name), 'w'))

    total_time = time.time() - tic
    print("total time taken: {}".format(total_time))
    json.dump([total_time], open('{}/time_taken.json'.format(folder_name), 'w'))
    torch.save(baseline_model.state_dict(), "{}/final_baseline.pt".format(folder_name))
    return all_train_losses, all_train_accuracies, all_test_losses, all_test_accuracies


def baseline(baseline_num_epochs, labelled_trainloader, testloader, folder_name,
             baseline_lr):
    """
    Interface between external files/calling functions and baseline training
    :param baseline_num_epochs: Number of epochs for training
    :param labelled_trainloader: Dataloader object for training
    :param testloader: Dataloader object for testing
    :param folder_name: Folder where loss and accuracy values are stored
    :param baseline_lr: Learning rate
    :return: train losses list, train accuracy list, test loss list and test accuracy list
    """
    baseline = BaseLine(len(classes))
    baseline = baseline.to(device)

    optimizer = optim.Adam(baseline.parameters(), lr=baseline_lr)

    criterion = nn.CrossEntropyLoss()

    all_train_losses, all_train_accuracies, all_test_losses, all_test_accuracies = baseline_wrapper(baseline,
                                                                                                     baseline_num_epochs,
                                                                                                     labelled_trainloader,
                                                                                                     testloader,
                                                                                                     optimizer,
                                                                                                     criterion,
                                                                                                     folder_name)
    return all_train_losses, all_train_accuracies, all_test_losses, all_test_accuracies


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_num_epochs", type=int)
    parser.add_argument("--baseline_lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--percentage_labelled")

    args = parser.parse_args()
    baseline_learning_rate = float(args.baseline_lr)
    batch_size = int(args.batch_size)
    percentage_labelled = float(args.percentage_labelled)

    print("Starting {} {}".format(percentage_labelled, 0.0))

    labelled_trainloader, _, testset, testloader = get_data(percentage_labelled, 0.0, batch_size)

    folder_name = '{}_{}_runs'.format(percentage_labelled, "0.0")
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    folder_name = "{}/baseline".format(folder_name)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    baseline_num_epochs = args.baseline_num_epochs
    all_train_losses, all_train_accuracies, all_test_losses, all_test_accuracies = baseline(baseline_num_epochs,
                                                                                            labelled_trainloader,
                                                                                            testloader,
                                                                                            folder_name,
                                                                                            baseline_learning_rate)
