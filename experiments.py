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
argparser.add_argument("--method", type=str, default='request', choices=['day', 'request'])
argparser.add_argument("--minor", type=int, default=0, help="select minor antigens")
argparser.add_argument("--alpha", default=0.001, type=float, help="learning rate")
argparser.add_argument("--nn", default=[128, 128, 128], type=int, nargs='+', help="layer sizes")
argparser.add_argument("--epsilon", default=1, type=float, help="exploration rate")
argparser.add_argument("--decay", default=1, type=float, help="epsilon decay")
argparser.add_argument("--episodes", default=20, type=int, help="number of episodes")
argparser.add_argument("--target", default=True, help="use a target network")
argparser.add_argument("--frequency", default=100, help="update frequency of target network")
argparser.add_argument("--buffer", default=500, help="experience replay buffer size")
args = argparser.parse_args()


def main():
    startTime = datetime.now()
    SETTINGS = Settings(method=args.method, minor=args.minor, alpha=args.alpha, n_neurons=args.nn, epsilon=args.epsilon,
                        decay=args.decay, episodes=args.episodes, target=args.target, frequency=args.frequency,
                        buffer=args.buffer)
    PARAMS = Params(SETTINGS)

    paths = [
        "results", f"results/{SETTINGS.model_name}",
        f"results/{SETTINGS.model_name}/a{SETTINGS.alpha}_g{SETTINGS.gamma}_b{SETTINGS.batch_size}",
        f"results/{SETTINGS.model_name}/a{SETTINGS.alpha}_g{SETTINGS.gamma}_b{SETTINGS.batch_size}/{SETTINGS.architecture}",
        f"results/{SETTINGS.model_name}/a{SETTINGS.alpha}_g{SETTINGS.gamma}_b{SETTINGS.batch_size}/{SETTINGS.architecture}/e{SETTINGS.epsilon}_target_{SETTINGS.target}_freq_{SETTINGS.target_frequency}_exp_{SETTINGS.buffer_size}/",
        "models", f"models/{SETTINGS.model_name}",
        f"models/{SETTINGS.model_name}/a{SETTINGS.alpha}_g{SETTINGS.gamma}_b{SETTINGS.batch_size}",
        f"models/{SETTINGS.model_name}/a{SETTINGS.alpha}_g{SETTINGS.gamma}_b{SETTINGS.batch_size}/{SETTINGS.architecture}",
        f"models/{SETTINGS.model_name}/a{SETTINGS.alpha}_g{SETTINGS.gamma}_b{SETTINGS.batch_size}/{SETTINGS.architecture}/e{SETTINGS.epsilon}_target_{SETTINGS.target}_freq_{SETTINGS.target_frequency}_exp_{SETTINGS.buffer_size}/"]

    for path in paths:
        SETTINGS.check_dir_existence(SETTINGS.home_dir + path)

    episodes = [*range(0, SETTINGS.episodes[1], 1)]
    random.shuffle(episodes)
    splits = [episodes[i:i + 5] for i in range(0, len(episodes), 5)]

    # Use k-fold cross validation to train and evaluate the agent
    for index in range(0, len(splits)):
        train_episodes = splits[:index] + splits[index + 1:]
        test_episodes = splits[index]

        print("CREATING ENVIRONMENT")
        env = MatchingEnv(SETTINGS, PARAMS)
        print("CREATING DQN")
        dqn = DQN(SETTINGS, env)

        print(f"alpha: {SETTINGS.alpha}, gamma: {SETTINGS.gamma}, batch size: {SETTINGS.batch_size}.")

        dqn.train(SETTINGS, PARAMS, train_episodes)
        dqn.test(SETTINGS, PARAMS, test_episodes)

    print(datetime.now() - startTime)


if __name__ == "__main__":
    main()
