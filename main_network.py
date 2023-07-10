from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import numpy as np
import os

from train_network import train, test

# Training settings
parser = argparse.ArgumentParser(description='NN settings')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--mps', action='store_true', default=False,
                    help='enables macOS GPU training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')


class Q_net(nn.Module):

    def __init__(self, input, output, nn):
        super().__init__()
        self.input = [input]
        self.output = [output]
        self.nn = nn
        self.model = self.build_nn()

    def build_nn(self):
        input_size = self.input
        output = self.output
        layer_sizes = input_size + self.nn + output

        assert len(layer_sizes) > 1
        layers = []
        for index in range(len(layer_sizes) - 1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index + 1])
            act = nn.ReLU() if index < len(layer_sizes) - 2 else nn.Identity()
            layers += (linear, act)
        return nn.Sequential(*layers)


class MyData(Dataset):
    def __init__(self, data_path, target_path, transform=None):

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
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.x[idx], self.y[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.x)


if __name__ == '__main__':
    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    kwargs = {'batch_size': args.batch_size,
              'shuffle': True}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       })

    torch.manual_seed(args.seed)
    mp.set_start_method('spawn', force=True)

    model = Q_net(72, 8, [128, 128]).model
    model.to(device)
    model.share_memory()  # gradients are allocated lazily, so they are not shared here

    processes = []

    data_path = 'NN training data/reg_ABDCcEeKkFyaFybJkaJkbMNSs/states/'
    target_path = 'NN training data/reg_ABDCcEeKkFyaFybJkaJkbMNSs/q_matrices/'

    dataset = MyData(data_path, target_path)

    train_size = int(0.75 * len(dataset))
    test_size = (len(dataset) - train_size)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_size = (len(train_dataset) - len(test_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

    args = parser.parse_args()

    train(0, args, model, device, train_dataset, kwargs)

    # for rank in range(args.num_processes):
    #     p = mp.Process(target=train, args=(rank, args, model, device,
    #                                        train_dataset, kwargs))
    #     # We first train the model across `num_processes` processes
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()

    # Once training is complete, we can test the model
    test(args, model, device, test_dataset, kwargs)
