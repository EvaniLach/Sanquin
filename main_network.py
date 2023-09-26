from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, TensorDataset, DataLoader, Subset
from datetime import datetime
from sklearn.model_selection import train_test_split

import numpy as np
import os

from train_network import train

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
parser.add_argument('--seed', type=int, default=99, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model_interval', type=int, default=10,
                    help='interval for saving nn weights')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
import platform


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
        self.layer_2 = nn.Linear(512, 256)
        self.layer_3 = nn.Linear(256, 128)
        self.layer_4 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.batchnorm4 = nn.BatchNorm1d(64)

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

        x = self.layer_4(x)
        x = self.batchnorm4(x)
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

    # Use 10% of total training data for testing model
    subset_indices, _ = train_test_split(
        range(len(dataset)),
        stratify=dataset.y,
        train_size=0.1,
        random_state=args.seed
    )

    # Split 80/20 for training and validation
    train_set, val_set = train_test_split(
        subset_indices,
        stratify=dataset.y[subset_indices],
        test_size=0.2,
        random_state=args.seed
    )

    val_split = TensorDataset(normalize(dataset.x[val_set]), dataset.y[val_set])
    train_split = TensorDataset(normalize(cap_outliers(dataset.x[train_set])), dataset.y[train_set])

    return train_split, val_split, dataset.y[train_set]


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


def cap_outliers(matrix):
    feature_indices = []
    for i in range(matrix.shape[1]):
        if i % 3 == 0:
            feature_indices.append(i)
            feature_indices.append(i + 1)

    for c in feature_indices:
        low = torch.quantile(matrix[:, c], 0.1)
        high = torch.quantile(matrix[:, c], 0.99)
        for r in range(matrix.shape[0]):
            if matrix[r, c] < low:
                matrix[r, c] = low
            elif matrix[r, c] > high:
                matrix[r, c] = high

    return matrix


if __name__ == '__main__':
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    # mp.set_start_method('spawn', force=True)

    # model = Q_net(24, 8, [128, 64]).model
    # .to(device)
    # model.share_memory()

    model = MulticlassClassification(num_feature=24, num_class=8)
    model.to(device)

    # train_dataset, val_dataset, test_dataset, targets = get_data()

    train_dataset, val_dataset, targets = get_data()

    args = parser.parse_args()

    print("Start training")
    startTime = datetime.now()

    train(0, args, model, device, train_dataset, targets, val_dataset)

    print(datetime.now() - startTime)

    # test(args, model, device, test_dataset)
