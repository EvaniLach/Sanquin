import time
import numpy as np
import pandas as pd

# Create the dataframe with all required columns, to store outputs during the simulations.
def initialize_output_dataframe(SETTINGS, PARAMS, hospital, episode):

    ##########
    # PARAMS #
    ##########

    antigens = PARAMS.major + PARAMS.minor
    ABOD_names = PARAMS.ABOD
    patgroups = PARAMS.patgroups
    ethnicities = ["Caucasian", "African", "Asian"]
    days = [i for i in range(SETTINGS.init_days + SETTINGS.test_days)]

    ##########
    # HEADER #
    ##########

    # General information.
    header = ["logged", "day", "location", "model name", "supply scenario", "demand scenario", "avg daily demand", "inventory size", "test days", "init days"]

    # Gurobi optimizer info.
    # header += ["gurobi status", "nvars", "calc time"]
    # header += ["objval shortages", "objval mismatches", "objval substitution", "objval fifo", "objval usability"]

    # RL stats.
    header += ["epsilon start", "epsilon decay", "epsilon min", "epsilon current", "learning rate", "gamma", "batch size"]
    
    # Information about patients, donors, demand and supply.
    # header += ["num patients"] + [f"num {eth} patients" for eth in ethnicities]
    # header += [f"num {pg} patients" for pg in patgroups]
    header += ["num units requested"]
    # header += [f"num units requested {pg}" for pg in patgroups]
    header += ["num supplied products"] + [f"num supplied {major}" for major in ABOD_names] + [f"num requests {major}" for major in ABOD_names]
    header += [f"num {major} in inventory" for major in ABOD_names]

    # Only if the offline model is used.
    # if self.line == "off":
    #     header += ["products available today", "products in inventory today"]

    # Which products were issued to which patiens.
    # header += ["avg issuing age"]
    # header += [f"{major0} to {major1}" for major0 in ABOD_names for major1 in ABOD_names]
    # header += [f"{eth0} to {eth1}" for eth0 in ethnicities for eth1 in ethnicities]
    # header += [f"num allocated at dc {pg}" for pg in patgroups]

    # Matching performance.
    header += ["reward", "issued but nonexistent", "issued but discarded"]
    header += ["num outdates", "num shortages"]
    # header += [f"num outdates {major}" for major in ABOD_names] + [f"num shortages {major}" for major in ABOD_names]
    header += [f"num mismatches {ag}" for ag in antigens]
    # header += [f"num shortages {pg}" for pg in patgroups] + [f"num {pg} {major+1} units short" for pg in patgroups for major in range(4)] + ["num unavoidable shortages"]
    # header += [f"num mismatches {pg} {ag}" for pg in patgroups for ag in antigens] + [f"num mismatched units {pg} {ag}" for pg in patgroups for ag in antigens]
    # header += [f"num mismatches {eth} {ag}" for eth in ethnicities for ag in antigens]

    df = pd.DataFrame(columns = header)

    # Set the dataframe's index to each combination of day and location name.

    df.loc[:,"day"] = days
    df.loc[:,"location"] = hospital.name

    df = df.set_index(['day'])
    df = df.fillna(0)

    ##################
    # ADD BASIC INFO #
    ##################

    df.loc[:,"logged"] = False
    df.loc[:,"model name"] = SETTINGS.model_name
    df.loc[:,"test days"] = SETTINGS.test_days
    df.loc[:,"init days"] = SETTINGS.init_days
    df.loc[:,"supply scenario"] = f"cau{round(SETTINGS.donor_eth_distr[0]*100)}_afr{round(SETTINGS.donor_eth_distr[1]*100)}_asi{round(SETTINGS.donor_eth_distr[2]*100)}_{episode}"
    
    # df.loc[:,"epsilon start"] = SETTINGS.epsilon
    df.loc[:,"epsilon decay"] = SETTINGS.epsilon_decay
    df.loc[:,"epsilon min"] = SETTINGS.epsilon_min
    df.loc[:,"learning rate"] = SETTINGS.alpha
    df.loc[:,"gamma"] = SETTINGS.gamma
    df.loc[:,"batch size"] = SETTINGS.batch_size

    df.loc[:,"demand scenario"] = f"{hospital.htype}_{episode}"
    df.loc[:,"avg daily demand"] = hospital.avg_daily_demand
    df.loc[:,"inventory size"] = hospital.inventory_size

    
    return df