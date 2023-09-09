from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from datetime import datetime
from sklearn.model_selection import train_test_split

import numpy as np
import os

from train_network import train, test

# Training settings
parser = argparse.ArgumentParser(description='NN settings')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.0004, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--seed', type=int, default=20, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model_interval', type=int, default=20,
                    help='interval for saving nn weights')

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


class MyData(Dataset):
    def __init__(self, data_path=None, target_path=None):
        train_list = []
        for i in os.listdir(data_path):
            data = np.load(data_path + i, allow_pickle=True)
            train_list.append(data)

        test_list = []
        for i in os.listdir(target_path):
            data = np.load(target_path + i, allow_pickle=True)
            test_list.append(data)

        self.x = torch.from_numpy(np.vstack(train_list)).float()
        self.y = torch.from_numpy(np.vstack(test_list)).float()

    def __getitem__(self, idx):
        sample = self.x[idx], self.y[idx]
        return sample

    def __len__(self):
        return len(self.x)


def get_data():
    # dir = 'C:/Users/evani/OneDrive/AI leiden/Sanquin/NN training data/'
    data_path = 'NN training data/1_1/states/'
    target_path = 'NN training data/1_1/q_matrices/'

    dataset = MyData(data_path, target_path)

    train_size = int(0.85 * len(dataset))
    test_size = (len(dataset) - train_size)

    # Split 0.85 of indices for initial train portion
    train_indices, test_indices, _, _ = train_test_split(
        range(len(dataset)),
        dataset.y,
        stratify=dataset.y,
        test_size=test_size,
    )

    # Save target value in train set to calculate class weights later on
    train_targets = dataset[train_indices][1]

    train_split = Subset(dataset, train_indices)
    test_slit = Subset(dataset, test_indices)

    # Split again to get 0.7 train and 0.15 validation sets
    train_indices, val_indices, _, _ = train_test_split(
        range(len(train_indices)),
        train_targets,
        stratify=train_targets,
        test_size=test_size,
    )

    val_split = Subset(dataset, val_indices)

    return train_split, val_split, test_slit, train_targets


if __name__ == '__main__':
    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda:5")
    else:
        device = torch.device("cpu")

    kwargs = {'batch_size': args.batch_size,
              'shuffle': True}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       })

    torch.manual_seed(args.seed)
    mp.set_start_method('spawn', force=True)

    model = Q_net(24, 8, [104, 43, 111]).model
    model.to(device)
    model.share_memory()

    processes = []

    train_dataset, val_dataset, test_dataset, targets = get_data()

    args = parser.parse_args()

    print("Start training")
    startTime = datetime.now()

    train(0, args, model, device, train_dataset, targets, kwargs)

    print(datetime.now() - startTime)

    test(args, model, device, test_dataset, kwargs)
