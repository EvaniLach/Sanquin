import pandas as pd
import pickle

from blood import *

class Distribution_center():
    
    def __init__(self, SETTINGS, e, supply_index = 0):

        # Name for the distribution center (currently not used)
        self.name = f"dc_{e}"

        # Read the supply that was generated using SETTINGS.mode = "supply"
        self.supply_data = pd.read_csv(SETTINGS.home_dir + f"supply/{SETTINGS.supply_size}/cau{round(SETTINGS.donor_eth_distr[0]*100)}_afr{round(SETTINGS.donor_eth_distr[1]*100)}_asi{round(SETTINGS.donor_eth_distr[2]*100)}_{e}.csv")
        
        # Keep track of the supply index to know which item of the supply data to read next.
        self.supply_index = supply_index


    # Read the required number of products from the supply data and add these products to the distribution center's inventory.
    def sample_supply_single_day(self, PARAMS, len_I, n_products, age = 0):

        # Select the next part of the supply scenario.
        data = self.supply_data.iloc[self.supply_index : self.supply_index + n_products]
        self.supply_index += n_products

        # Transform the newly received supply, as read from the data file, to a blood group index,
        # and increase the number of products of that blood group in stock by 1.
        supply = [0] * len_I
        for i in data.index:
            supply[vector_to_bloodgroup_index(data.loc[i, PARAMS.major + PARAMS.minor])] += 1

        return supply


    def pickle(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)