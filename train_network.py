import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd


def train_epoch(epoch, model, args, device, train_loader, optimizer, weights):
    model.train()
    running_loss = 0.0
    loss = nn.CrossEntropyLoss(weight=weights)

    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = loss(output, target.to(device))
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        correct += (output == target).float().sum()

    accuracy = (correct / len(train_loader.dataset)) * 100
    print('Train Epoch: {} \tLoss: {:.6f} \tAccuracy: {:.6f}'.format(
        epoch, running_loss, accuracy))

    return running_loss


def calculate_weights(targets):
    class_probs = torch.sum(targets, dim=0) / len(targets)
    class_weights = 1 / class_probs
    return class_weights


def train(rank, args, model, device, train_dataset, targets, dataloader_kwargs):
    torch.manual_seed(args.seed)

    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_values = []
    weights = calculate_weights(targets)

    for epoch in range(1, args.epochs + 1):
        loss_values.append(train_epoch(epoch, model, args, device, train_loader, optimizer, weights))
        if epoch % args.model_interval == 0:
            torch.save(model.state_dict(), 'models/kickstart/{}/model_{}.pt'.format(args.seed, epoch))

    df = pd.DataFrame(loss_values, columns=["loss"])
    df.to_csv('results/kickstart/{}/loss_1.csv'.format(args.seed), index=False)


def test(args, model, device, test_dataset, dataloader_kwargs):
    torch.manual_seed(args.seed)

    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_kwargs)

    test_epoch(model, device, test_loader)


def test_epoch(model, device, data_loader):
    model.eval()
    test_loss = 0
    accuracy = 0
    loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            test_loss += loss(output, target.to(device), reduction='sum').item()
            for i in range(len(output)):
                if torch.argmax(output[i]) == torch.argmax(target[i]):
                    accuracy += 1

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(
        test_loss))
    print('\nTest set: Accuracy: {:.4f}.\n'
          '\n[{}/{}]\n'.format(accuracy / len(data_loader.dataset), accuracy, len(data_loader.dataset)))


def validate(model, val_loader, device):
    model.eval()
    val_loss = 0
    accuracy = 0

    loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.view(data.size(0), -1).to(device), target.to(device)
            output = model(data)
            val_loss += loss(output, target.to(device), reduction='sum').item()
            for i in range(len(output)):
                if torch.argmax(output[i]) == torch.argmax(target[i]):
                    accuracy += 1

    val_loss /= len(val_loader.dataset)
    print('\nVal set: Average loss: {:.4f}\n'.format(
        val_loss))
    print('\nVal set: Accuracy: {:.4f}.\n'
          '\n[{}/{}]\n'.format(accuracy / len(val_loader.dataset), accuracy, len(val_loader.dataset)))
