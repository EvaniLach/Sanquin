from smac import HyperparameterOptimizationFacade as HPO, Scenario, RunHistory
from ConfigSpace import ConfigurationSpace, Configuration, UniformFloatHyperparameter, OrdinalHyperparameter, \
    UniformIntegerHyperparameter, GreaterThanCondition, CategoricalHyperparameter, EqualsCondition

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from env import *
from dqn_torch import *

from settings import *
from params import *
import numpy as np


def make_config_space():
    cs = ConfigurationSpace(seed=0)
    n_layers = cs.add_hyperparameter(OrdinalHyperparameter('n_layers', [1, 2, 3, 4]))
    n_neurons1 = cs.add_hyperparameter(UniformIntegerHyperparameter('n_neurons1', lower=16, upper=256, q=16))
    n_neurons2 = cs.add_hyperparameter(UniformIntegerHyperparameter('n_neurons2', lower=16, upper=256, q=16))
    n_neurons3 = cs.add_hyperparameter(UniformIntegerHyperparameter('n_neurons3', lower=16, upper=256, q=16))
    n_neurons4 = cs.add_hyperparameter(UniformIntegerHyperparameter('n_neurons4', lower=16, upper=512, q=2))

    cs.add_condition(GreaterThanCondition(n_neurons2, n_layers, 1))
    cs.add_condition(GreaterThanCondition(n_neurons3, n_layers, 2))
    cs.add_condition(GreaterThanCondition(n_neurons4, n_layers, 3))

    cs.add_hyperparameter(
        CategoricalHyperparameter('optimizer', ['adam']))
    cs.add_hyperparameter(
        UniformFloatHyperparameter('optimizer_lr', lower=1e-5, upper=1e-1, log=True))

    cs.add_hyperparameter(CategoricalHyperparameter('activation', ['relu', 'tanh', 'elu', 'gelu']))
    return cs


def run_a(config, seed=20):
    print(config)
    n_neurons = [config['n_neurons1']]

    if config['n_layers'] > 1:
        n_neurons.append(config['n_neurons2'])
    if config['n_layers'] > 2:
        n_neurons.append(config['n_neurons3'])
    if config['n_layers'] > 3:
        n_neurons.append(config['n_neurons4'])

    SETTINGS = Settings(0)
    PARAMS = Params(SETTINGS)
    print("CREATING ENVIRONMENT")
    env = MatchingEnv(SETTINGS, PARAMS)
    print("CREATING DQN")
    dqn = DQN(SETTINGS, env, n_neurons=n_neurons,
              activation=config['activation'], alpha=config['optimizer_lr'])
    reward = dqn.train(SETTINGS, PARAMS)
    print(reward)
    return np.mean(reward)



def run(config, budget):
    sum = 0
    with ProcessPoolExecutor() as executor:
        for res in executor.map(run_a, repeat(config, 10)):
            sum += res
        return 200 - sum / 10.0


if __name__ == '__main__':
    cs = make_config_space()
    scenario = Scenario(cs, output_directory=r"/home/s1949624/Sanquin/",
                        n_trials=1,
                        seed=10)
    # load/import classes
    from dask.distributed import Client
    import logging

    # set up cluster and workers
    # client = Client(n_workers=15,
    #                 threads_per_worker=1,
    #                 memory_limit='512GB',
    #                 silence_logs=logging.ERROR)
    # intensifier = Hyperband


    smac = HPO(scenario, run_a, overwrite=True)
    incumbent = smac.optimize()
    print(incumbent)
