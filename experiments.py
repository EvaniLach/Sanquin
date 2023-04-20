from env import *
from dqn import *

from settings import *
from params import *
import argparse
from supply import *
from demand import *

# NOTES
# We currently assume that each requested unit is a separate request.
# Compatibility now only on major antigens -> specify to include patient group specific mandatory combinations.

argparser = argparse.ArgumentParser()
argparser.add_argument("--method", type=str, choices=['day', 'request'])
argparser.add_argument("--minor", type=int, default=0, help="select minor antigens")
argparser.add_argument("--dev", default=None, help="GPU ID to use")
args = argparser.parse_args()

def main(method, n_neurons, alpha, minor):

    SETTINGS = Settings(method, n_neurons, alpha, minor)
    PARAMS = Params(SETTINGS)

    paths = [
        "results", f"results/{SETTINGS.model_name}", f"results/{SETTINGS.model_name}/a{SETTINGS.alpha}_g{SETTINGS.gamma}_b{SETTINGS.batch_size}_{SETTINGS.minor}",
        "models", f"models/{SETTINGS.model_name}", f"models/{SETTINGS.model_name}/a{SETTINGS.alpha}_g{SETTINGS.gamma}_b{SETTINGS.batch_size}_{SETTINGS.minor}"]
    for path in paths:
        SETTINGS.check_dir_existence(SETTINGS.home_dir + path)

    print("CREATING ENVIRONMENT")
    env = MatchingEnv(SETTINGS, PARAMS)
    print("CREATING DQN")
    dqn = DQN(SETTINGS, env)

    print(f"alpha: {SETTINGS.alpha}, gamma: {SETTINGS.gamma}, batch size: {SETTINGS.batch_size}.")

    # Train the agent
    dqn.train(SETTINGS, PARAMS)
    # test comment

if __name__ == "__main__":
    n_neurons = [64, 32, 16]
    alphas = [0.01, 0.001, 0.001, 0.0001, 0.00001]

    for alpha in alphas:
        main(args.method, n_neurons, alpha, args.minor)