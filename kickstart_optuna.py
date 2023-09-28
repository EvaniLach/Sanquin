import os

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import argparse
import numpy as np

from main_network import Q_net, MyData

DEVICE = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
INPUT = 1 * 24
OUTPUT = 8
DIR = os.getcwd()
EPOCHS = 50

N_TRAIN_BATCHES = 500
N_VALID_BATCHES = 100

parser = argparse.ArgumentParser(description='NN settings')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--seed', type=int, default=20, metavar='N',
                    help='seed')
args = parser.parse_args()


def define_model(trial):
    n_layers = trial.suggest_int("n_layers", 1, 5)
    layers = []

    in_features = INPUT
    p = trial.suggest_float("dropout", 0.1, 0.8)
    for i in range(n_layers):
        neurons = trial.suggest_int("n_units_l{}".format(i), 4, 512)
        out_features = neurons
        act = nn.ReLU()
        linear = nn.Linear(in_features, out_features)
        layers += (linear, act)
        layers.append(nn.BatchNorm1d(neurons))
        in_features = out_features
        if i > 1:
            layers.append(nn.Dropout(p))

    layers.append((nn.Linear(in_features, OUTPUT)))

    return nn.Sequential(*layers)


def get_data():
    # dir = 'C:/Users/evani/OneDrive/AI leiden/Sanquin/NN training data/'
    data_path = 'NN training data/1_1/states/'
    target_path = 'NN training data/1_1/q_matrices/'

    dataset = MyData(data_path, target_path)

    # # Use 10% of total training data for testing model
    # subset_indices, _ = train_test_split(
    #     range(len(dataset)),
    #     stratify=dataset.y,
    #     train_size=0.1,
    #     random_state=args.seed
    # )

    # Split 80/20 for training and validation
    train_set, val_set = train_test_split(
        range(len(dataset)),
        stratify=dataset.y,
        test_size=0.2,
        random_state=args.seed
    )

    val_split = TensorDataset(normalize(dataset.x[val_set]), dataset.y[val_set])
    train_split = TensorDataset(normalize(dataset.x[train_set]), dataset.y[train_set])

    return train_split, val_split, dataset.y[train_set]


def normalize(matrix):
    matrix.to(DEVICE)
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


def cap_outliers(matrix):
    feature_indices = []

    for i in range(matrix.shape[1]):
        if i % 3 == 0:
            feature_indices.append(i)
            feature_indices.append(i + 1)

    for c in feature_indices:
        low = torch.quantile(matrix[:, c], 0.1)
        high = torch.quantile(matrix[:, c], 0.99)
        for r in range(matrix.shape[0]):
            if matrix[r, c] < low:
                matrix[r, c] = low
            elif matrix[r, c] > high:
                matrix[r, c] = high

    return matrix


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


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    correct_pred = (y_pred_tags == y_test).float()

    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)
    return acc


def objective(trial, train, val, targets):
    torch.manual_seed(args.seed)
    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = "Adam"
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    sampler, cw = weighted_sampler(targets)

    train_loader = DataLoader(train, batch_size=65, sampler=sampler)
    val_loader = DataLoader(val, batch_size=64)

    loss = nn.CrossEntropyLoss()

    val_loss = 0
    val_acc = 0

    # Training of the model.
    for epoch in range(EPOCHS):
        print("epoch", epoch)
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= N_TRAIN_BATCHES:
                break
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            train_loss = loss(output, target)
            train_loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.inference_mode():
            for batch_idx, (data, target) in enumerate(val_loader):
                if batch_idx >= N_VALID_BATCHES:
                    break
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)

                val_loss += loss(output, target).item()

                batch_acc = multi_acc(output, target)
                val_acc += batch_acc.item()

        val_loss = val_loss / N_VALID_BATCHES
        val_acc = val_acc / N_VALID_BATCHES

        trial.report(val_acc, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_acc


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    train, val, targets = get_data()

    study.optimize(objective(train, val, targets), n_trials=100, timeout=None)

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
