import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils import *
from pretrainer import *
from finetuner import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device is", device)


# def train_autoencoder_wrapper(auto_encoder_model, num_epochs, unlabelled_trainloader, testset, optimizer,
#                               criterion, folder_name):
#     tic = time.time()
#     all_losses = []
#     for epoch in tqdm(range(num_epochs), leave=False):
#         loss, auto_encoder_model = train_model(auto_encoder_model, unlabelled_trainloader, optimizer, criterion)
#         tqdm.write("epoch: {} train loss: {} time elapsed: {}".format(epoch, loss, time.time() - tic))
#         all_losses.append(loss)
#
#         subset_indices = random.sample(range(0, len(testset)), 10)
#         subset = torch.utils.data.Subset(testset, subset_indices)
#         testloader_subset = torch.utils.data.DataLoader(subset, batch_size=1, num_workers=0, shuffle=False)
#
#         for i, data in enumerate(testloader_subset):
#             original_data = data[0][0].permute(1, 2, 0)
#             original_data = (original_data * 0.5) + 0.5
#             plt.imshow(original_data)
#             plt.savefig('{}/{}_{}_original'.format(folder_name, i, epoch))
#             reconstructed_image = auto_encoder_model(data[0].to(device))[0].detach().cpu()
#             reconstructed_image = reconstructed_image.permute(1, 2, 0)
#             reconstructed_image = (reconstructed_image * 0.5) + 0.5
#             plt.imshow(reconstructed_image)
#             plt.savefig('{}/{}_{}_reconstructed'.format(folder_name, i, epoch))
#         json.dump(all_losses, open("{}_auto_encoder_loss.json".format(folder_name), 'w'))
#         torch.save(auto_encoder_model.state_dict(), "{}/epoch_{}.pt".format(folder_name, epoch))
#     return all_losses

#
# def train(num_epochs, unlabelled_trainloader, testset):
#     auto_encoder_model = Autoencoder()
#     auto_encoder_model = auto_encoder_model.to(device)
#     criterion = nn.MSELoss()
#     lr = 0.001
#     optimizer = optim.Adam(auto_encoder_model.parameters(), lr=lr)
#     folder_name = '{}_{}_runs'.format(percentage_labelled, percentage_unlabelled)
#     if not os.path.exists(folder_name):
#         os.mkdir(folder_name)
#
#     folder_name = "{}/finetuner".format(folder_name)
#     if not os.path.exists(folder_name):
#         os.mkdir(folder_name)
#
#     all_losses = train_autoencoder_wrapper(auto_encoder_model, num_epochs, unlabelled_trainloader, testset, optimizer,
#                                            criterion, folder_name)
#     json.dump(all_losses, open("{}_auto_encoder_loss.json".format(folder_name), 'w'))
#     return auto_encoder_model, all_losses


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

    auto_encoder_model, all_losses = pretrain(pretrainer_num_epochs, unlabelled_trainloader, testset)

    all_train_losses, all_train_accuracies, all_test_losses, all_test_accuracies = \
        finetune(auto_encoder_model, finetuner_num_epochs, labelled_trainloader, testloader)




