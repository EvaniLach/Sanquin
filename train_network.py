import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd


def train_epoch(epoch, model, args, device, train_loader, optimizer):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    torch.save(model.state_dict(), 'models/kickstart/{}/model_{}'.format(args.seed, epoch))
    return running_loss


def train(rank, args, model, device, train_dataset, dataloader_kwargs):
    torch.manual_seed(args.seed)

    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_values = []

    for epoch in range(1, args.epochs + 1):
        loss_values.append(train_epoch(epoch, model, args, device, train_loader, optimizer))

    df = pd.DataFrame(loss_values, columns=["loss"])
    df.to_csv('results/kickstart/{}/loss_1.csv'.format(args.seed), index=False)


def test(args, model, device, test_dataset, dataloader_kwargs):
    torch.manual_seed(args.seed)

    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_kwargs)

    test_epoch(model, device, test_loader)


def test_epoch(model, device, data_loader):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            test_loss += F.mse_loss(output, target.to(device), reduction='sum').item()  # sum up batch loss

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(
        test_loss))
