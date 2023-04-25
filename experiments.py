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
argparser.add_argument("--method", type=str, default='request',choices=['day', 'request'])
argparser.add_argument("--minor", type=int, default=0, help="select minor antigens")
argparser.add_argument("--dev", default=None, help="GPU ID to use")
args = argparser.parse_args()

def main(method, alpha, minor):
    print(minor)
    SETTINGS = Settings(method, alpha, minor)
    PARAMS = Params(SETTINGS)

    paths = [
        "results", f"results/{SETTINGS.model_name}", f"results/{SETTINGS.model_name}/learning_rates",
        f"results/{SETTINGS.model_name}/learning_rates/a{SETTINGS.alpha}",
        "models", f"models/{SETTINGS.model_name}", f"models/{SETTINGS.model_name}/learning_rates",
        f"models/{SETTINGS.model_name}/learning_rates/a{SETTINGS.alpha}"]
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
   # n_neurons = [64, 32, 16]
    alphas = [0.01, 0.001, 0.001, 0.0001, 0.00001]

    for alpha in alphas:
        main(method=args.method, alpha=alpha, minor=args.minor)