from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, TensorDataset, DataLoader, Subset
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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
parser.add_argument('--seed', type=int, default=20, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model_interval', type=int, default=10,
                    help='interval for saving nn weights')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
args = parser.parse_args()


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
        nn.init.kaiming_normal_(self.layer_1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.layer_1.bias)
        self.layer_2 = nn.Linear(512, 256)
        nn.init.kaiming_normal_(self.layer_2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.layer_2.bias)
        self.layer_3 = nn.Linear(256, 128)
        nn.init.kaiming_normal_(self.layer_3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.layer_3.bias)
        self.layer_4 = nn.Linear(128, 64)
        nn.init.kaiming_normal_(self.layer_4.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.layer_4.bias)
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
        for i in sorted(os.listdir(data_path)):
            data = np.load(data_path + i, allow_pickle=True)
            x_list.append(data)

        y_list = []
        for i in sorted(os.listdir(target_path)):
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
    data_path = 'NN training data/35_8/states/'
    target_path = 'NN training data/35_8/q_matrices/'

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

    scaler = MinMaxScaler()
    scaler.fit(dataset.x[train_set])

    val_split = TensorDataset(normalize(dataset.x[val_set], scaler), dataset.y[val_set])
    train_split = TensorDataset(normalize(dataset.x[train_set], scaler), dataset.y[train_set])

    return train_split, val_split, dataset.y[train_set]


def normalize(matrix, scaler):
    scaled = scaler.transform(matrix.numpy())

    return torch.from_numpy(scaled)


if __name__ == '__main__':
    args = parser.parse_args()

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print("DEVICE: ", device)

    torch.manual_seed(args.seed)
    # mp.set_start_method('spawn', force=True)

    # model = Q_net(24, 8, [128, 64]).model
    # .to(device)
    # model.share_memory()

    model = MulticlassClassification(num_feature=8 * 43, num_class=8)
    model.to(device)

    # train_dataset, val_dataset, test_dataset, targets = get_data()

    train_dataset, val_dataset, targets = get_data()

    args = parser.parse_args()

    print("Start training")
    startTime = datetime.now()

    train(0, args, model, device, train_dataset, targets, val_dataset)

    print(datetime.now() - startTime)

    # test(args, model, device, test_dataset)
