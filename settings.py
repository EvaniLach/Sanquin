import pandas as pd
import os

class Settings():

    def __init__(self, method, minor, alpha, nn, epsilon, ed):

        # Working directory.
        # self.home_dir = r"C:/Users/evani/OneDrive/AI leiden/Sanquin/RL_matching-main/"
        self.home_dir = r"/home/s1949624/RL_matching/"

        # Output files will be stored in directory results/[model_name].
        if method == 'day':
            self.model_name = "daily_scratch"       # matching per day, training from scratch
        # self.model_name = "daily_guided"        # matching per day, training guided by MINRAR strategy
        else:
            self.model_name = "request_scratch"     # matching per request, training from scratch
        # self.model_name = "request_guided"      # matching per request, training guided by MINRAR strategy
        self.method = method

        #########################
        # SIMULATION PARAMETERS #
        #########################

        # Only the results of test days will be logged.
        self.test_days = round(2.5 * 365)
        self.init_days = 2 * 35

        # (x,y): Episode numbers range(x,y) will be optimized.
        # The total number of simulations executed will thus be y - x.
        self.episodes = (0,25)

        # Number of hospitals considered. If more than 1 (regional and university combined), a distribution center is included.
        # "regional": Use the patient group distribution of the OLVG, a regional hospital, with average daily demand of 50 products.
        # "university": Use the patient group distribution of the AMC, a university hospital, with average daily demand of 100 products.
        self.n_hospitals = {
            "regional" : 1,
            "university" : 0,
            # "manual" : 0,
        }

        self.avg_daily_demand = {
            "regional" : 50,
            "university" : 100,
            # "manual" : 10,
        }

        # Size factor for distribution center and hospitals.
        # Average daily demand x size factor = inventory size.
        self.inv_size_factor_dc = 6         # CHANGE (no doubt)
        self.inv_size_factor_hosp = 3

        # "major": Only match on the major antigens.
        # "relimm": Use relative immunogenicity weights for mismatching.
        # "patgroups": Use patient group specific mismatching weights.
        self.strategy = "major"
        self.patgroup_musts = True

        # Which minor antigens to use
        self.minor = minor

        ##############################
        # GENERATING DEMAND / SUPPLY #
        ##############################

        self.donor_eth_distr = [1, 0, 0]  # [Caucasian, African, Asian]
        
        if sum(self.n_hospitals.values()) > 1:
            self.supply_size = (self.init_days + self.test_days) * self.inv_size_factor_dc * sum([self.n_hospitals[htype] * self.avg_daily_demand[htype] for htype in self.n_hospitals.keys()])
        else:
            self.supply_size = (self.init_days + self.test_days) * self.inv_size_factor_hosp * sum([self.n_hospitals[htype] * self.avg_daily_demand[htype] for htype in self.n_hospitals.keys()])


        ##########################
        # REINFORCEMENT LEARNING #
        ##########################

        # "train" for training the RL model
        # "test" for running simulations with saved model
        self.RL_mode = "train"

        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = ed
        self.alpha = alpha         # learning rate
        self.gamma = 0.5           # discount factor
        self.batch_size = 50       # batch size for replay buffer
        self.nn = nn

        ####################
        # GUROBI OPTIMIZER #
        ####################

        self.show_gurobi_output = False     # True or False
        self.gurobi_threads = None          # Number of threads available, or None in case of no limit
        self.gurobi_timeout = None          # Number of minutes allowed for optimization, None in case of no limit


    # Generate a file name for exporting log or result files.
    def generate_filename(self, SETTINGS, output_type, e):

        path = self.home_dir + f"/{output_type}/{self.model_name}/"
        path += f"a{self.alpha}_g{self.gamma}_b{self.batch_size}/"
        path += f"{SETTINGS.nn}/"
        path += f"{self.method}_{e}"

        return path

    # Check whether a given path exists, and create the path if it doesn't.
    def check_dir_existence(self, path):
        if os.path.exists(path) == False:
            os.mkdir(path)
