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
argparser.add_argument("--alpha", default=0.001, help="learning rate")
argparser.add_argument("--nn", default=[64], type=int, nargs='+', help="layer sizes")
args = argparser.parse_args()

def main():

    SETTINGS = Settings(method=args.method, minor=args.minor, alpha=args.alpha, nn=args.nn)
    PARAMS = Params(SETTINGS)

    paths = [
        "results", f"results/{SETTINGS.model_name}", f"results/{SETTINGS.model_name}/architectures",
        f"results/{SETTINGS.model_name}/architectures/{SETTINGS.nn}",
        "models", f"models/{SETTINGS.model_name}", f"models/{SETTINGS.model_name}/architectures",
        f"models/{SETTINGS.model_name}/architectures/{SETTINGS.nn}"]
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
    main()