import torch.nn as nn
from torch import optim
from tqdm import tqdm
import torch
import numpy as np
from matplotlib import pyplot as plt
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device is", device)


def train_autoencoder_wrapper(auto_encoder_model, num_epochs, unlabelled_trainloader, testloader, optimizer,
                              criterion, folder_name):
    tic = time.time()
    all_losses = []
    all_test_losses = []
    prev_loss = np.inf
    for epoch in tqdm(range(num_epochs), leave=False):
        loss, _, auto_encoder_model = train_model(auto_encoder_model, unlabelled_trainloader, optimizer, criterion)
        test_loss = evaluate_autoencoder(auto_encoder_model, testloader, criterion)
        tqdm.write("epoch: {} train loss: {} test loss: {} time elapsed: {}".format(epoch, loss, test_loss,
                                                                                    time.time() - tic))
        all_losses.append(loss)
        all_test_losses.append(test_loss)

        # subset_indices = random.sample(range(0, len(testset)), 10)
        # subset = torch.utils.data.Subset(testset, subset_indices)
        # testloader_subset = torch.utils.data.DataLoader(subset, batch_size=1, num_workers=0, shuffle=False)

        # for i, data in enumerate(testloader_subset):
        #     original_data = data[0][0].permute(1, 2, 0)
        #     original_data = (original_data * 0.5) + 0.5
        #     plt.imshow(original_data)
        #     plt.savefig('{}/test_images/{}_{}_original'.format(folder_name, i, epoch))
        #     reconstructed_image = auto_encoder_model(data[0].to(device))[0].detach().cpu()
        #     reconstructed_image = reconstructed_image.permute(1, 2, 0)
        #     reconstructed_image = (reconstructed_image * 0.5) + 0.5
        #     plt.imshow(reconstructed_image)
        #     plt.savefig('{}/test_images/{}_{}_reconstructed'.format(folder_name, i, epoch))
        json.dump(all_losses, open("{}/epoch_{}_loss.json".format(folder_name, epoch), 'w'))
        json.dump(all_test_losses, open("{}/epoch_{}_test_loss.json".format(folder_name, epoch), 'w'))
        if loss < prev_loss:
            prev_loss = loss
            torch.save(auto_encoder_model.state_dict(), "{}/pretrainer.pt".format(folder_name))

    total_time = time.time() - tic
    print("total time taken: {}".format(total_time))
    json.dump([total_time], open('{}/time_taken.json'.format(folder_name), 'w'))
    torch.save(auto_encoder_model.state_dict(), "{}/final_pretrainer.pt".format(folder_name))
    return all_losses, all_test_losses


def pretrain(num_epochs, unlabelled_trainloader, testloader, folder_name, lr):
    auto_encoder_model = Autoencoder()
    auto_encoder_model = auto_encoder_model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(auto_encoder_model.parameters(), lr=lr)

    all_losses, all_test_losses = train_autoencoder_wrapper(auto_encoder_model, num_epochs, unlabelled_trainloader,
                                                            testloader, optimizer, criterion, folder_name)
    json.dump(all_losses, open("{}_auto_encoder_loss.json".format(folder_name), 'w'))
    json.dump(all_test_losses, open("{}_auto_encoder_test_loss.json".format(folder_name), 'w'))
    return auto_encoder_model, all_losses, all_test_losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrainer_num_epochs", type=int)
    parser.add_argument("--finetuner_num_epochs", type=int)
    parser.add_argument("--percentage_labelled", type=float)
    parser.add_argument("--percentage_unlabelled", type=float)
    parser.add_argument("--pretrainer_lr", type=float)
    parser.add_argument("--batch_size", type=int)

    args = parser.parse_args()

    pretrainer_num_epochs = args.pretrainer_num_epochs
    finetuner_num_epochs = args.finetuner_num_epochs
    percentage_labelled = float(args.percentage_labelled)
    percentage_unlabelled = float(args.percentage_unlabelled)
    learning_rate = float(args.pretrainer_lr)
    batch_size = int(args.batch_size)

    folder_name = '{}_{}_runs'.format(percentage_labelled, percentage_unlabelled)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    folder_name = "{}/pretrainer".format(folder_name)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    labelled_trainloader, unlabelled_trainloader, testset, testloader = get_data(percentage_labelled,
                                                                                 percentage_unlabelled,
                                                                                 batch_size)

    auto_encoder_model, all_losses, all_test_losses = pretrain(pretrainer_num_epochs, unlabelled_trainloader,
                                                               testloader, folder_name, learning_rate)
