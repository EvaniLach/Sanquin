import os

import optuna
import pandas as pd
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import argparse

from main_network import Q_net, MyData

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT = 1 * 72
OUTPUT = 8
DIR = os.getcwd()
EPOCHS = 50
BATCHSIZE = 64

N_TRAIN_EXAMPLES = BATCHSIZE * 2000
N_VALID_EXAMPLES = BATCHSIZE * 500

parser = argparse.ArgumentParser(description='NN settings')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
torch.cuda.empty_cache()


def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 4)
    layers = []

    in_features = INPUT
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        act = nn.ReLU()
        linear = nn.Linear(in_features, out_features)
        layers += (linear, act)
        # p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        # layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append((nn.Linear(in_features, OUTPUT), nn.Softmax()))

    return nn.Sequential(*layers)


def get_data():
    data_path = 'NN training data/reg_ABDCcEeKkFyaFybJkaJkbMNSs/states/'
    target_path = 'NN training data/reg_ABDCcEeKkFyaFybJkaJkbMNSs/q_matrices/'

    dataset = MyData(data_path, target_path)

    train_size = int(0.85 * len(dataset))
    test_size = (len(dataset) - train_size)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_size = (len(train_dataset) - len(test_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

    return train_dataset, val_dataset


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

    train_dataset, val_dataset = get_data()

    train_loader = torch.utils.data.DataLoader(train_dataset, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)

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
            train_loss = F.cross_entropy(output, target)
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
                val_loss += F.cross_entropy(output, target)

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
