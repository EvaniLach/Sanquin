from __future__ import print_function
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.sampler import Sampler

from main_network import MyData

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


def test(model, device, test_dataset):
    torch.manual_seed(args.seed)

    test_loader = DataLoader(test_dataset, batch_size=64)

    test_epoch(model, device, test_loader)


def test_epoch(model, device, data_loader):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0

    loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            batch_loss = loss(output, target)
            batch_acc = multi_acc(output, target)

            epoch_loss += batch_loss.item()
            epoch_acc += batch_acc.item()

    rel_loss = epoch_loss / len(data_loader)
    rel_acc = epoch_acc / len(data_loader)

    print('Test_loss: {:.4f} \tTest_acc: {:.2f}'.format(
        rel_loss, rel_acc))


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # path = 'C:/Users/evani/OneDrive/AI leiden/Sanquin/Results/kickstart/'
    path = 'models/kickstart/4/'

    model = MulticlassClassification(num_feature=24, num_class=8)
    model.load_state_dict(torch.load(
        path + 'model_20.pt'))
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
    test(model, device, test_dataset)
