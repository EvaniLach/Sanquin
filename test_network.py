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


class Q_net(nn.Module):

    def __init__(self, input, output, nn, p=0):
        super().__init__()
        self.input = input
        self.output = output
        self.nn = nn
        self.p = p
        self.model = self.define_model()

    def define_model(self):
        layers = []

        in_features = self.input
        for i in range(len(self.nn) - 1):
            out_features = self.nn[i]
            act = nn.ReLU()
            linear = nn.Linear(in_features, out_features)
            layers += (linear, act)
            # layers.append(nn.Dropout(self.p[i]))
            in_features = out_features
        layers.append(nn.Linear(in_features, self.output))

        return nn.Sequential(*layers)


if __name__ == '__main__':
    network = Q_net(72, 8, [64, 64])
    network.model.load_state_dict(torch.load('models/kickstart/1/model_25'))
