import pandas as pd
import pickle
from bitstring import BitArray

from blood import *

class Hospital():
    
    def __init__(self, SETTINGS, htype, e):

        self.htype = htype                                                              # Hospital type ("regional" or "university")
        self.name = f"{htype[:3]}_{e}"                                                  # Name for the hospital.
        self.avg_daily_demand = SETTINGS.avg_daily_demand[htype]                        # Average daily number of units requested within this hospital.
        self.inventory_size = SETTINGS.inv_size_factor_hosp * self.avg_daily_demand     # Size of the hospital's inventory.

        # Read the demand that was generated using SETTINGS.mode = "demand".
        self.demand_data = pd.read_csv(SETTINGS.home_dir + f"demand/{self.avg_daily_demand}/{SETTINGS.test_days + SETTINGS.init_days}/{htype}_{e}.csv")


    def sample_requests_single_day(self, PARAMS, R_shape, day = 0):

        # Select the part of the demand scenario belonging to the given day.
        # TODO: checken of het klopt met "Day Available" en "Day Needed"
        data = self.demand_data.loc[self.demand_data["Day Available"] == day]

        # Transform the new requests, as read from the data file, to a blood group index and a lead time.
        requests = np.zeros([R_shape[0], R_shape[1]])
        for i in data.index:
            lead_time = data.loc[i,"Day Needed"] - day - 1
            requests[BitArray(data.loc[i, PARAMS.major + PARAMS.minor]).uint, lead_time] += data.loc[i, "Num Units"]

        return requests


    def pickle(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
