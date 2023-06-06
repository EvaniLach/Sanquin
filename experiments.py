from env import *
from dqn_torch import *

from settings import *
from params import *
import argparse
from supply import *
from demand import *
from datetime import datetime

# NOTES
# We currently assume that each requested unit is a separate request.
# Compatibility now only on major antigens -> specify to include patient group specific mandatory combinations.

argparser = argparse.ArgumentParser()
argparser.add_argument("--method", type=str, default='request',choices=['day', 'request'])
argparser.add_argument("--minor", type=int, default=0, help="select minor antigens")
argparser.add_argument("--alpha", default=0.001, type=float, help="learning rate")
argparser.add_argument("--nn", default=[64, 64], type=int, nargs='+', help="layer sizes")
argparser.add_argument("--epsilon", default=0.15, type=float, help="exploration rate")
argparser.add_argument("--decay", default=1, type=float, help="epsilon decay")
argparser.add_argument("--episodes", default=25, type=int, help="number of episodes")
argparser.add_argument("--target", default=False, help="use a target network")
argparser.add_argument("--frequency", default=250, help="update frequency of target network")
args = argparser.parse_args()

def main():
    startTime = datetime.now()
    SETTINGS = Settings(method=args.method, minor=args.minor, alpha=args.alpha, n_neurons=args.nn, epsilon=args.epsilon,
                        decay=args.decay, episodes=args.episodes, target=args.target, frequency=args.frequency)
    PARAMS = Params(SETTINGS)

    paths = [
        "results", f"results/{SETTINGS.model_name}", f"results/{SETTINGS.model_name}/a{SETTINGS.alpha}_g{SETTINGS.gamma}_b{SETTINGS.batch_size}",
        f"results/{SETTINGS.model_name}/a{SETTINGS.alpha}_g{SETTINGS.gamma}_b{SETTINGS.batch_size}/{SETTINGS.nn}",
        f"results/{SETTINGS.model_name}/a{SETTINGS.alpha}_g{SETTINGS.gamma}_b{SETTINGS.batch_size}/{SETTINGS.nn}/e{SETTINGS.epsilon}",
        f"results/{SETTINGS.model_name}/a{SETTINGS.alpha}_g{SETTINGS.gamma}_b{SETTINGS.batch_size}/{SETTINGS.nn}/e{SETTINGS.epsilon}/target_{SETTINGS.target}/",
        f"results/{SETTINGS.model_name}/a{SETTINGS.alpha}_g{SETTINGS.gamma}_b{SETTINGS.batch_size}/{SETTINGS.nn}/e{SETTINGS.epsilon}/target_{SETTINGS.target}/target_{SETTINGS.target_frequency}/",
        "models", f"models/{SETTINGS.model_name}", f"models/{SETTINGS.model_name}/a{SETTINGS.alpha}_g{SETTINGS.gamma}_b{SETTINGS.batch_size}",
        f"models/{SETTINGS.model_name}/a{SETTINGS.alpha}_g{SETTINGS.gamma}_b{SETTINGS.batch_size}/{SETTINGS.nn}",
        f"models/{SETTINGS.model_name}/a{SETTINGS.alpha}_g{SETTINGS.gamma}_b{SETTINGS.batch_size}/{SETTINGS.nn}/e{SETTINGS.epsilon}",
        f"models/{SETTINGS.model_name}/a{SETTINGS.alpha}_g{SETTINGS.gamma}_b{SETTINGS.batch_size}/{SETTINGS.nn}/e{SETTINGS.epsilon}/target_{SETTINGS.target}/",
        f"models/{SETTINGS.model_name}/a{SETTINGS.alpha}_g{SETTINGS.gamma}_b{SETTINGS.batch_size}/{SETTINGS.nn}/e{SETTINGS.epsilon}/target_{SETTINGS.target}/target_{SETTINGS.target_frequency}/"]
    for path in paths:
        SETTINGS.check_dir_existence(SETTINGS.home_dir + path)

    print("CREATING ENVIRONMENT")
    env = MatchingEnv(SETTINGS, PARAMS)
    print("CREATING DQN")
    dqn = DQN(SETTINGS, env)

    print(f"alpha: {SETTINGS.alpha}, gamma: {SETTINGS.gamma}, batch size: {SETTINGS.batch_size}.")

    # Train the agent
    dqn.train(SETTINGS, PARAMS)
    print(datetime.now() - startTime)

if __name__ == "__main__":
    main()
