import torch
from utils import *
from pretrainer import *
from finetuner import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device is", device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrainer_num_epochs", type=int)
    parser.add_argument("--finetuner_num_epochs", type=int)
    parser.add_argument("--percentage_labelled", type=float)
    parser.add_argument("--percentage_unlabelled", type=float)
    args = parser.parse_args()

    pretrainer_num_epochs = args.pretrainer_num_epochs
    finetuner_num_epochs = args.finetuner_num_epochs
    percentage_labelled = float(args.percentage_labelled)
    percentage_unlabelled = float(args.percentage_unlabelled)

    labelled_trainloader, unlabelled_trainloader, testset, testloader = get_data(percentage_labelled,
                                                                                 percentage_unlabelled)

    base_folder_name = '{}_{}_runs'.format(percentage_labelled, percentage_unlabelled)

    if not os.path.exists(base_folder_name):
        os.mkdir(base_folder_name)

    pretrainer_folder_name = "{}/pretrainer".format(base_folder_name)
    if not os.path.exists(pretrainer_folder_name):
        os.mkdir(pretrainer_folder_name)

    finetuner_folder_name = "{}/finetuner".format(base_folder_name)
    if not os.path.exists(finetuner_folder_name):
        os.mkdir(finetuner_folder_name)

    auto_encoder_model, all_losses = pretrain(pretrainer_num_epochs, unlabelled_trainloader, testset,
                                              pretrainer_folder_name)

    all_train_losses, all_train_accuracies, all_test_losses, all_test_accuracies = \
        finetune(auto_encoder_model, finetuner_num_epochs, labelled_trainloader, testloader, finetuner_folder_name)




