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
    prev_loss = np.inf
    stored_accuracy = False
    for epoch in tqdm(range(num_epochs), leave=False):
        loss, train_accuracy, finetuner_model = train_model(finetuner_model, labelled_trainloader, optimizer,
                                                            criterion, classification=True)
        all_train_losses.append(loss)
        all_train_accuracies.append(train_accuracy)
        json.dump(all_train_losses, open("{}/epoch_{}_loss.json".format(folder_name, epoch), 'w'))
        json.dump(all_train_accuracies, open("{}/epoch_{}_accuracy.json".format(folder_name, epoch), 'w'))
        test_loss, test_accuracy = evaluate_classification(finetuner_model, testloader, criterion)
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
            torch.save(finetuner_model.state_dict(), "{}/finetuner.pt".format(folder_name))

        if test_accuracy > 0.40 and not stored_accuracy:
            stored_accuracy = True
            json.dump([time.time() - tic, epoch], open("{}/time_to_accuracy.json".format(folder_name), 'w'))

    total_time = time.time() - tic
    print("total time taken: {}".format(total_time))
    json.dump([total_time], open('{}/time_taken.json'.format(folder_name), 'w'))
    torch.save(finetuner_model.state_dict(), "{}/final_finetuner.pt".format(folder_name))
    return all_train_losses, all_train_accuracies, all_test_losses, all_test_accuracies


def finetune(auto_encoder_model, finetuner_num_epochs, labelled_trainloader, testloader, folder_name,
             finetuning_lr, pretraining_lr):
    finetuner = FineTuner(auto_encoder_model, len(classes))
    finetuner = finetuner.to(device)

    my_list = [x[0] for x in auto_encoder_model.named_parameters() if 'encoder' in x]
    params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in my_list, finetuner.named_parameters()))))
    base_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in my_list, finetuner.named_parameters()))))

    optimizer = optim.Adam([{'params': base_params}, {'params': params, 'lr': pretraining_lr}], lr=finetuning_lr)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam([{'params': finetuner.parameters(), 'lr': finetuning_lr},
    #                         {'params': auto_encoder_model.parameters(), 'lr': pretraining_lr}])

    all_train_losses, all_train_accuracies, all_test_losses, all_test_accuracies = finetuner_wrapper(finetuner,
                                                                                                     finetuner_num_epochs,
                                                                                                     labelled_trainloader,
                                                                                                     testloader,
                                                                                                     optimizer,
                                                                                                     criterion,
                                                                                                     folder_name)
    return all_train_losses, all_train_accuracies, all_test_losses, all_test_accuracies


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--finetuner_num_epochs", type=int)
    parser.add_argument("--percentage_labelled", type=float)
    parser.add_argument("--percentage_unlabelled", type=float)
    parser.add_argument("--finetuner_lr", type=float)
    parser.add_argument("--pretrainer_lr", type=float)
    parser.add_argument("--batch_size", type=int)

    args = parser.parse_args()
    percentage_labelled = float(args.percentage_labelled)
    percentage_unlabelled = float(args.percentage_unlabelled)
    finetuner_learning_rate = float(args.finetuner_lr)
    pretrainer_learning_rate = float(args.pretrainer_lr)
    batch_size = int(args.batch_size)

    labelled_trainloader, unlabelled_trainloader, testset, testloader = get_data(percentage_labelled,
                                                                                 percentage_unlabelled,
                                                                                 batch_size)

    folder_name = '{}_{}_runs'.format(percentage_labelled, percentage_unlabelled)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    folder_name = "{}/finetuner".format(folder_name)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    pretrained_model_path = args.pretrained_model_path
    finetuner_num_epochs = args.finetuner_num_epochs
    auto_encoder_model = load_pretrained_model(pretrained_model_path)
    all_train_losses, all_train_accuracies, all_test_losses, all_test_accuracies = finetune(auto_encoder_model,
                                                                                            finetuner_num_epochs,
                                                                                            labelled_trainloader,
                                                                                            testloader,
                                                                                            folder_name,
                                                                                            finetuner_learning_rate,
                                                                                            pretrainer_learning_rate)
