import os

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

import argparse
import numpy as np

from main_network import Q_net, MyData

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT = 1 * 24
OUTPUT = 8
DIR = os.getcwd()
EPOCHS = 50
BATCHSIZE = 64

N_TRAIN_EXAMPLES = BATCHSIZE * 2000
N_VALID_EXAMPLES = BATCHSIZE * 500

parser = argparse.ArgumentParser(description='NN settings')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')


def define_model(trial):
    n_layers = trial.suggest_int("n_layers", 1, 6)
    layers = []

    in_features = INPUT
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        act = nn.ReLU()
        linear = nn.Linear(in_features, out_features)
        layers += (linear, act)
        in_features = out_features
    layers.append((nn.Linear(in_features, OUTPUT)))

    return nn.Sequential(*layers)


# def get_data():
#     dir = 'C:/Users/evani/OneDrive/AI leiden/Sanquin/NN training data/'
#     data_path = dir + 'NN training data/reg_ABDCcEeKkFyaFybJkaJkbMNSs/1_1 backup/states/'
#     target_path = dir + 'NN training data/reg_ABDCcEeKkFyaFybJkaJkbMNSs/1_1 backup/q_matrices/'
#
#     dataset = MyData(data_path, target_path)
#
#     train_size = int(0.85 * len(dataset))
#     test_size = (len(dataset) - train_size)
#
#     train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
#
#     train_size = (len(train_dataset) - len(test_dataset))
#     train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])
#
#     return train_dataset, val_dataset

def get_data():
    # dir = 'C:/Users/evani/OneDrive/AI leiden/Sanquin/NN training data/'
    data_path = 'NN training data/1_1/states/'
    target_path = 'NN training data/1_1/q_matrices/'

    dataset = MyData(data_path, target_path)

    train_size = int(0.85 * len(dataset))
    test_size = (len(dataset) - train_size)

    # Split 0.85 of indices for initial train portion
    train_indices, test_indices, _, _ = train_test_split(
        range(len(dataset)),
        dataset.y,
        stratify=dataset.y,
        test_size=test_size,
    )

    # Save target value in train set to calculate class weights later on
    train_targets = dataset[train_indices][1]

    # Split again to get 0.7 train and 0.15 validation sets
    train_indices, val_indices, _, _ = train_test_split(
        range(len(train_indices)),
        train_targets,
        stratify=train_targets,
        test_size=test_size,
    )

    print(dataset[val_indices][0][0])

    val_split = Subset(normalize(dataset[val_indices][0]), val_indices)
    train_split = Subset(normalize(dataset[train_indices][0]), train_indices)

    return train_split, val_split, train_targets


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


def objective(trial):
    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = "Adam"
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    args = parser.parse_args()
    kwargs = {'batch_size': args.batch_size,
              'shuffle': True}

    train_dataset, val_dataset, train_targets = get_data()

    train_loader = DataLoader(train_dataset, **kwargs)
    val_loader = DataLoader(val_dataset, **kwargs)

    class_probs = torch.sum(train_targets, dim=0) / len(train_dataset)
    class_weights = 1 / class_probs

    loss = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    if torch.cuda.is_available():
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       })

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break
            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            train_loss = loss(output, target)
            train_loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                output = model(data)
                val_loss += loss(output, target)

        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, timeout=None)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
