import gym
import numpy as np
import collections
import pickle

from action_to_matches import *
from hospital import *
from dc import *

# Define the environment for the reinforcement learning algorithm
class MatchingEnv(gym.Env):
    def __init__(self, SETTINGS, PARAMS):

        self.num_bloodgroups = 2**len(PARAMS.major + PARAMS.minor)

        # DAY-BASED
        # Each state is a matrix of 2**len(antigens) × (35 + 8)
        # Vertical: all considered blood groups, each row index representing the integer representation of a blood group.
        # Horizontal: product age (0,1,...,34) + number of days until issuing (6,5,...,0)
        # Each cell contains the number of requests/products of that type
        if SETTINGS.method == 'day':
            self.state = np.zeros([self.num_bloodgroups, PARAMS.max_age + PARAMS.max_lead_time])
            # Set hospital type to 'regional', since we are only using this one for now
            I_size = SETTINGS.inv_size_factor_hosp * SETTINGS.avg_daily_demand[0]
            # Each action is a matrix of 2**len(antigens) × inventory size
            # Vertical: all considered blood groups
            # Horizontal: number of products issued from that blood group (possibly in binary notation)
            self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.num_bloodgroups, I_size))                     # real number, one-hot encoded

        # REQUEST-BASED
        # Each state is a matrix of 2**len(antigens) × (35 + 8 + 1)
        # Vertical: all considered blood groups, each row index representing the integer representation of a blood group.
        # Horizontal: product age (0,1,...,34) + number of days until issuing (7,6,...,0) + binary indicating which request is considered currently.
        # Each cell (except for the right-most column) contains the number of requests/products of that type.
        else:
            self.state = np.zeros([self.num_bloodgroups, PARAMS.max_age + PARAMS.max_lead_time + 1])
            # Each action is an array of len(antigens), representing the antigen profile of the issued product.
            # self.action_space = gym.spaces.MultiBinary(len(antigens))
            self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.num_bloodgroups,))
        
        # Current day in the simulation.
        self.day = 0


    # def save(self, SETTINGS, path):
    #     for p in path.split("/"):
    #         SETTINGS.check_dir_existence(p)
    #     with open(path + f"env.pickle", 'wb') as f:
    #         pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


    def reset(self, SETTINGS, PARAMS, e, htype):
        
        self.day = 0

        # Initialize the hospital. A distribution center is also initialized to provide the hospital with random supply.
        self.dc = Distribution_center(SETTINGS, e)
        self.hospital = Hospital(SETTINGS, htype, e)

        # Create the part of the state representing the inventory.
        I = np.zeros([self.num_bloodgroups, PARAMS.max_age])
        I[:,0] = self.dc.sample_supply_single_day(PARAMS, len(I), self.hospital.inventory_size)

        # Create the part of the state representing requests.
        R = self.hospital.sample_requests_single_day(PARAMS, [self.num_bloodgroups, PARAMS.max_lead_time], self.day)

        # REQUEST-BASED
        # Create a new column with 1 in the first row where the right-most column is > 0, and 0 otherwise.
        current_r = np.zeros((R.shape[0], 1))
        r = np.where(R[:,-1]>0)[0]
        if len(r) > 0:
            current_r[r[0]] = 1

        # self.state = np.concatenate((I, R), axis=1)  # DAY-BASED
        self.state = np.concatenate((I, R, current_r), axis=1)  # REQUEST-BASED

    # REQUEST-BASED
    def log_state(self, PARAMS, day, df):

        # List of all considered bloogroups in integer representation.
        bloodgroups = list(range(self.num_bloodgroups))

        # The inventory is represented by the left part of the state -> matrix of size |bloodgroups| × max age
        I = self.state[:,:PARAMS.max_age]
        R = self.state[:,PARAMS.max_age:-1]
        current_r = self.state[:,-1]
        
        # Create lists of blood groups in integer representation.
        inventory, requests_today = [], []
        for bg in bloodgroups:
            # All blood groups present in the inventory.
            inventory.extend([bg] * int(sum(I[bg])))

            # All requests from the state that need to be satisfied today.
            requests_today.extend([bg] * int(R[bg,-1]))
        df.loc[day,"num units requested"] = len(requests_today)
        df.loc[day,"num supplied products"] = sum(I[:,0])

        ABOD_names = PARAMS.ABOD

        major_bins = len(bloodgroups) / len(ABOD_names)
        for m in range(len(ABOD_names)):
            start = int(m * major_bins)
            end = int((m * major_bins) + major_bins)
            df.loc[day,f"num supplied {ABOD_names[m]}"] = sum(I[start:end, 0])
            df.loc[day,f"num requests {ABOD_names[m]}"] = sum(R[start:end, -1])
            df.loc[day,f"num {ABOD_names[m]} in inventory"] = sum(sum(I[start:end]))

    # REQUEST-BASED
    def calculate_reward(self, SETTINGS, PARAMS, action, day, df):

        ABOD_names = PARAMS.ABOD

        ######################
        ## STATE AND ACTION ##
        ######################

        # List of all considered bloogroups in integer representation.
        bloodgroups = list(range(self.num_bloodgroups))
        antigens = PARAMS.major + PARAMS.minor

        # The inventory is represented by the left part of the state -> matrix of size |bloodgroups| × max age
        I = self.state[:,:PARAMS.max_age]
        R = self.state[:,PARAMS.max_age:-1]
        r = int(np.argmax(self.state[:,-1]))
    
        ######################
        ## CALCULATE REWARD ##
        ######################

        reward = 0
        
        # If the product issued is actually present in the inventory..
        if sum(I[action]) > 0:

            # Remove the issued products from the inventory, where the oldest product is removed first.
            I[action, np.where(I[action] > 0)[0][-1]] -= 1

            comp = binarray(not_compatible(r, action))
            
            # The issued product is not compatible with the request -> shortage.
            if sum(comp[:3]) > 0:
                reward -= 10 + 1    
                df.loc[day,"num shortages"] += 1
                df.loc[day,"issued but discarded"] += 1                

            else:
                A = {antigens[k] : k for k in range(len(antigens))}
                A_no_Fyb = [ag for ag in A.keys() if ag != "Fyb"]

                # Retrieve the antigen (and patient group) weights.
                w = np.array(PARAMS.relimm_weights[antigens])[0]

                # Mismatch penalties.
                mismatch_penalties = 0
                for ag in A_no_Fyb:
                    mismatch_penalties += comp[A[ag]] * w[A[ag]]
                    df.loc[day, f"num mismatches {ag}"] += comp[A[ag]]
                if "Fyb" in antigens:
                    mismatch_penalties += comp[A["Fyb"]] * int(bin(r)[A["Fya"]+2]) * w[A["Fyb"]]
                    df.loc[day, f"num mismatches Fyb"] += comp[A["Fyb"]] * int(bin(r)[A["Fya"]+2])
                reward -= mismatch_penalties

        else:
            reward -= 50 + 10     # The issued product is not present in the inventory.
            df.loc[day,"issued but nonexistent"] += 1
            df.loc[day,"num shortages"] += 1

        num_outdates = sum(I[:,PARAMS.max_age-1])
        reward -= num_outdates                  # Penalty of 1 for each outdated product.
        df.loc[day,"num outdates"] += num_outdates

        df.loc[day,"reward"] += reward
        
        return reward, df

    # REQUEST-BASED
    def next_request(self, PARAMS):

        # Remove the already matched request.
        self.state[:,-2] -= self.state[:,-1]

        # Check if there are still requests for today to match.
        if sum(self.state[:,-2]) > 0:

            # Create a new column with 1 in the first row where the right-most column is > 0, and 0 otherwise.
            current_r = np.zeros(self.state.shape[0])
            current_r[np.where(self.state[:,-2]>0)[0][0]] = 1

            self.state[:,-1] = current_r

            return self.state, False

        else:

            # Proceed to the next day
            self.next_day(PARAMS)
            
            return self.state, True

    # REQUEST-BASED
    def next_day(self, PARAMS):
        
        # Increase the day count.
        self.day += 1

        I = self.state[:,:PARAMS.max_age]
        R = self.state[:,PARAMS.max_age:-1]

        # Increase the age of all products (also removing all outdated products).
        I[:,1:PARAMS.max_age] = I[:,:PARAMS.max_age-1]

        # Sample new supply to fill the inventory upto its maximum capacity.
        I[:,0] = self.dc.sample_supply_single_day(PARAMS, len(I), max(0, self.hospital.inventory_size - int(sum(sum(I)))))

        # Increase lead time of all other requests.
        R = np.insert(R[:, :-1], 0, values=0, axis=1)

        # Sample new requests
        R += self.hospital.sample_requests_single_day(PARAMS, R.shape, self.day)

         # Create a new column with 1 in the first row where the right-most column is > 0, and 0 otherwise.
        current_r = np.zeros((R.shape[0], 1))
        r = np.where(R[:,-1]>0)[0]
        if len(r) > 0:
            current_r[r[0]] = 1

        self.state = np.concatenate((I, R, current_r), axis=1)  # REQUEST-BASED  

    # DAY-BASED
    def step(self, SETTINGS, PARAMS, action, day, df):

        ABOD_names = PARAMS.ABOD

        ######################
        ## STATE AND ACTION ##
        ######################

        # List of all considered bloogroups in integer representation.
        bloodgroups = list(range(self.num_bloodgroups))

        # The inventory is represented by the left part of the state -> matrix of size |bloodgroups| × max age
        I = self.state[:,:PARAMS.max_age]
        R = self.state[:,PARAMS.max_age:]
        
        # Create lists of blood groups in integer representation.
        inventory, issued_action, requests_today = [], [], []
        for bg in bloodgroups:
            # All blood groups present in the inventory.
            inventory.extend([bg] * int(sum(I[bg])))

            # All inventory products from the action that should be issued today.
            # issued_action.extend([bg] * int(np.where(action[bg]==1)[0]))
            issued_action.extend([bg] * action[bg])

            # All requests from the state that need to be satisfied today.
            requests_today.extend([bg] * int(R[bg,-1]))

        df.loc[day,"num units requested"] = len(requests_today)
        df.loc[day,"num supplied products"] = sum(I[:,0])

        major_bins = len(bloodgroups) / len(ABOD_names)
        for m in range(len(ABOD_names)):
            start = int(m * major_bins)
            end = int((m * major_bins) + major_bins)
            df.loc[day,f"num supplied {ABOD_names[m]}"] = sum(I[start:end, 0])
            df.loc[day,f"num requests {ABOD_names[m]}"] = sum(R[start:end, -1])
            df.loc[day,f"num {ABOD_names[m]} in inventory"] = sum(sum(I[start:end]))

        #####################
        ## GET ASSIGNMENTS ##
        #####################
        
        # Divide the 'issued' list in two, depending on whether they are actually present in the inventory.
        inv = collections.Counter(inventory)
        iss = collections.Counter(issued_action)
        nonexistent = list((iss - inv).elements())      # Products attempted to issue, but not present in the inventory.
        issued = list((inv & iss).elements())           # Products issued and available for issuing.

        if len(requests_today) > 0:
            # Assign all issued products to today's requests, first minimizing shortages, then minimizing the mismatch penalty.
            shortages, mismatches, assigned, discarded, df = action_to_matches(PARAMS, issued, requests_today, day, df)
        else:
            shortages, mismatches, assigned = 0, 0, 0
            discarded = issued.copy()

        num_outdates = sum(I[:,PARAMS.max_age-1])

        ######################
        ## CALCULATE REWARD ##
        ######################

        reward = 0
        reward -= 50 * len(nonexistent)         # Penalty of 50 for each nonexistent product that was attempted to issue.
        reward -= 10 * shortages                # Penalty of 10 for each shortage.
        reward -= 5 * mismatches                # Penalty of 5 multiplied by the total mismatch penalty, which is weighted according to relative immunogenicity.
        reward -= num_outdates                  # Penalty of 1 for each outdated product.
        reward -= len(discarded)                # Penalty of 1 for each discarded product (issued but not assigned).

        df.loc[day,"reward"] = reward
        df.loc[day,"issued but nonexistent"] = len(nonexistent)
        df.loc[day,"num shortages"] = shortages
        df.loc[day,"num outdates"] = num_outdates
        df.loc[day,"issued but discarded"] = len(discarded)

        ######################
        ## GO TO NEXT STATE ##
        ######################

        # Increase the day count.
        self.day += 1

        # Remove all issued products from the inventory, where the oldest products are removed first.
        for bg in issued:
            I[bg, np.where(I[bg] > 0)[0][-1]] -= 1

        # Increase the age of all products (also removing all outdated products).
        I[:,1:PARAMS.max_age] = I[:,:PARAMS.max_age-1]

        # Return the number of products to be supplied, in order to fill the inventory upto its maximum capacity.
        I[:,0] = self.dc.sample_supply_single_day(PARAMS, len(I), max(0, self.hospital.inventory_size - int(sum(sum(I)))))

        # Remove all today's requests and increase lead time of all other requests.
        R = np.insert(R[:, :-1], 0, values=0, axis=1)

        # Sample new requests
        R += self.hospital.sample_requests_single_day(PARAMS, R.shape, self.day)

        # Update the state with the updated inventory and requests.
        self.state = np.concatenate((I, R), axis=1)

        return self.state, reward, self.day, df