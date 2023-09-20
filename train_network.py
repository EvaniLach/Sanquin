import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import pandas as pd

accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}


def train_epoch(epoch, model, args, device, train_loader, optimizer, weights):
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    loss = nn.CrossEntropyLoss(weight=weights.to(device))

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.to(device))

        batch_loss = loss(output, target)
        batch_acc = multi_acc(output, target)

        epoch_loss += batch_loss.item()
        epoch_acc += batch_acc.item()

        batch_loss.backward()
        optimizer.step()

    rel_loss = epoch_loss / len(train_loader)
    rel_acc = epoch_acc / len(train_loader)

    print('Train Epoch: {} \tTrain_loss: {:.4f} \tTrain_acc: {:.2f}'.format(
        epoch, rel_loss, rel_acc))

    return rel_loss, rel_acc


def validate(model, val_loader, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0

    loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            batch_loss = loss(output, target)
            batch_acc = multi_acc(output, target)

            epoch_loss += batch_loss.item()
            epoch_acc += batch_acc.item()

    rel_loss = epoch_loss / len(val_loader)
    rel_acc = epoch_acc / len(val_loader)

    print('\tVal_loss: {:.4f} \tVal_acc: {:.2f}'.format(
        rel_loss, rel_acc))

    return rel_loss, rel_acc


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    print(y_pred_tags)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    print(acc)
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


def train(rank, args, model, device, train_dataset, targets, val_dataset):
    torch.manual_seed(args.seed)
    sampler, cw = weighted_sampler(targets)

    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=64)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loss, train_acc = [], []
    val_loss, val_acc = [], []

    print("Val acc & loss: ", validate(model, val_loader, device))

    for epoch in range(1, args.epochs + 1):
        epoch_tloss, epoch_tacc = train_epoch(epoch, model, args, device, train_loader, optimizer, cw)
        epoch_vloss, epoch_vacc = validate(model, val_loader, device)

        train_loss.append(epoch_tloss), train_acc.append(epoch_tacc)
        val_loss.append(epoch_vloss), val_acc.append(epoch_vacc)

        if epoch % args.model_interval == 0:
            torch.save(model.state_dict(), 'models/kickstart/{}/model_{}.pt'.format(args.seed, epoch))

    train_df = pd.DataFrame(list(zip(train_loss, train_acc)), columns=["loss", "accuracy"])
    val_df = pd.DataFrame(list(zip(val_loss, val_acc)), columns=["loss", "accuracy"])

    train_df.to_csv('results/kickstart/{}/train_4.csv'.format(args.seed), index=False)
    val_df.to_csv('results/kickstart/{}/val_4.csv'.format(args.seed), index=False)


def test(args, model, device, test_dataset):
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
