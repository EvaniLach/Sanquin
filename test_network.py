from __future__ import print_function
import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler

from main_network import MyData
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='NN settings')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')


class Q_net(nn.Module):

    def __init__(self, input, output, nn):
        super().__init__()
        self.input = input
        self.output = output
        self.nn = nn
        # self.p = p
        self.model = self.define_model()

    def define_model(self):
        layers = []

        in_features = self.input
        for i in range(len(self.nn)):
            out_features = self.nn[i]
            act = nn.ReLU()
            linear = nn.Linear(in_features, out_features)
            layers += (linear, act)
            # layers.append(nn.Dropout(self.p[i]))
            in_features = out_features
        layers.append(nn.Linear(in_features, self.output))

        return nn.Sequential(*layers)


def test(args, model, device, test_dataset, dataloader_kwargs):
    torch.manual_seed(args.seed)

    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_kwargs)

    test_epoch(model, device, test_loader)


def test_epoch(model, device, data_loader):
    model.eval()
    test_loss = 0
    accuracy = 0

    with torch.no_grad():
        results = np.zeros((1, 2))
        for data, target in data_loader:
            output = model(data.to(device))
            for i in range(len(output)):
                results = np.vstack([results, [torch.argmax(target[i]), torch.argmax(output[i])]])

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(
        test_loss))
    print('\nTest set: Accuracy: {:.4f}.\n'
          '\n[{}/{}]\n'.format(accuracy / len(data_loader.dataset), accuracy, len(data_loader.dataset)))
    df_results = pd.DataFrame(results)
    df_results.to_csv('results/test/s20_[64_128_128__64]_1_1.csv')


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # path = 'C:/Users/evani/OneDrive/AI leiden/Sanquin/Results/kickstart/'
    path = 'models/kickstart/20/'

    model = Q_net(24, 8, [64, 128, 128, 64]).model
    model.load_state_dict(torch.load(
        path + 'model_500.pt'))
    model.to(device)
    model.share_memory()

    data_path = 'NN training data/1_1/states/'
    target_path = 'NN training data/1_1/q_matrices/'

    dataset = MyData(data_path, target_path)

    train_size = int(0.75 * len(dataset))
    test_size = (len(dataset) - train_size)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    args = parser.parse_args()
    kwargs = {'batch_size': args.batch_size,
              'shuffle': True}
    test(args, model, device, test_dataset, kwargs)
