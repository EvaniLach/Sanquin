from __future__ import print_function
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader

from main_network import MulticlassClassification, get_data

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


def test(model, device, test_dataset):
    torch.manual_seed(args.seed)
    test_loader = DataLoader(test_dataset, batch_size=64)
    test_epoch(model, device, test_loader)


def test_epoch(model, device, data_loader):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0

    torch.manual_seed(args.seed)

    loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            if batch_idx == 1:
                print(data)
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
    args = parser.parse_args()
    device = ('cpu' if torch.cuda.is_available() else 'cpu')

    # path = 'C:/Users/evani/OneDrive/AI leiden/Sanquin/Results/kickstart/'
    path = 'models/kickstart/20/'

    model = MulticlassClassification(num_feature=24, num_class=8)
    model.load_state_dict(torch.load(
        path + 'model_3.pt'))
    model.to(device)

    train_dataset, val_dataset, targets = get_data()

    kwargs = {'batch_size': args.batch_size,
              'shuffle': True}
    test(model, device, val_dataset)
