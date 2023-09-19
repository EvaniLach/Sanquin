import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import pandas as pd


def train_epoch(epoch, model, args, device, train_loader, optimizer, weights):
    model.train()
    running_loss = 0.0
    train_acc = 0.0
    loss = nn.CrossEntropyLoss(weight=weights.to(device))

    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.to(device))

        train_loss = loss(output, target.to(device))
        train_acc = multi_acc(output, target)

        running_loss += train_loss.item()
        train_acc += train_acc.item()

        train_loss.backward()
        optimizer.step()

    print('Train Epoch: {} \tLoss: {:.6f} \tAccuracy: {:.6f}'.format(
        epoch, running_loss, train_acc))

    return running_loss


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


def inv_weights(targets):
    class_probs = torch.sum(targets, dim=0) / len(targets)
    class_weights = 1 / class_probs
    return class_weights


def weighted_sampler(targets):
    class_counts = torch.unique(targets, return_counts=True)[1]
    class_weights = 1. / class_counts
    class_weights_all = class_weights[targets]

    sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
    )

    return sampler, class_weights


def train(rank, args, model, device, train_dataset, targets):
    torch.manual_seed(args.seed)
    sampler, cw = weighted_sampler(targets)

    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_values = []

    for epoch in range(1, args.epochs + 1):
        loss_values.append(train_epoch(epoch, model, args, device, train_loader, optimizer, cw))
        if epoch % args.model_interval == 0:
            torch.save(model.state_dict(), 'models/kickstart/{}/model_{}.pt'.format(args.seed, epoch))

    df = pd.DataFrame(loss_values, columns=["loss"])
    df.to_csv('results/kickstart/{}/loss_1.csv'.format(args.seed), index=False)


def test(args, model, device, test_dataset):
    torch.manual_seed(args.seed)

    test_loader = DataLoader(test_dataset, batch_size=64)

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
