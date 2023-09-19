from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, TensorDataset, DataLoader
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
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
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


class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x


class MyData(Dataset):
    def __init__(self, data_path=None, target_path=None):
        x_list = []
        for i in os.listdir(data_path):
            data = np.load(data_path + i, allow_pickle=True)
            x_list.append(data)

        y_list = []
        for i in os.listdir(target_path):
            data = np.load(target_path + i, allow_pickle=True)
            y_list.append(data)

        self.x = torch.from_numpy(np.vstack(x_list)).float()
        self.y = torch.from_numpy(np.vstack(y_list)).float()
        self.y = (self.y == 1).nonzero()[:, 1]

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

    train_targets = dataset[train_indices][1]

    # Split again to get 0.7 train and 0.15 validation sets
    train_indices, val_indices, _, _ = train_test_split(
        range(len(train_indices)),
        train_targets,
        stratify=train_targets,
        test_size=test_size,
    )

    # Save target value in train set to calculate class weights later on
    train_targets = dataset[train_indices][1]

    test_split = TensorDataset(normalize(dataset[test_indices][0]), dataset[test_indices][1])
    val_split = TensorDataset(normalize(dataset[val_indices][0]), dataset[val_indices][1])
    train_split = TensorDataset(normalize(dataset[train_indices][0]), dataset[train_indices][1])

    return train_split, val_split, test_split, train_targets


def normalize(matrix):
    columns = matrix.shape[1]
    feature_indices = [(i, i + 1) for i in range(columns) if (i % 3 == 0)]
    min_max = []

    for i in feature_indices:
        min_max.append((torch.min(matrix[:, i[0]]), torch.max(matrix[:, i[0]])))
        min_max.append((torch.min(matrix[:, i[1]]), torch.max(matrix[:, i[1]])))

    for i in range(len(matrix)):
        index = 0
        for j in feature_indices:
            matrix[i, j[0]] = (
                    (matrix[i, j[0]] - min_max[index][0]) / (min_max[index][1] - min_max[index][0]))
            matrix[i, j[1]] = (
                    (matrix[i, j[1]] - min_max[index + 1][0]) / (min_max[index + 1][1] - min_max[index + 1][0]))
            index += 2
    return matrix


if __name__ == '__main__':
    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed)
    mp.set_start_method('spawn', force=True)

    # model = Q_net(24, 8, [64, 128, 128, 64]).model
    # model.to(device)
    # model.share_memory()

    model = MulticlassClassification(num_feature=24, num_class=8)
    model.to(device)

    train_dataset, val_dataset, test_dataset, targets = get_data()

    args = parser.parse_args()

    print("Start training")
    startTime = datetime.now()

    train(0, args, model, device, train_dataset, targets)

    print(datetime.now() - startTime)

    test(args, model, device, test_dataset)
