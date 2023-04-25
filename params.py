import pandas as pd

class Params():
    
    def __init__(self, SETTINGS):


        ####################
        # BLOOD PARAMETERS #
        ####################

        # Shelf life of the blood products in inventory.
        self.max_age = 35           # The age at which inventory products expire, so the maximum age to be issued = 34. 
        self.max_lead_time = 8      # Days 0,1,...,7.

        # Major blood groups, major antigens, and minor antigens.
        self.ABOD =  {0:"O-", 1:"O+", 2:"B-", 3:"B+", 4:"A-", 5:"A+", 6:"AB-", 7:"AB+"}
        self.major = ["A", "B", "D"]
        # self.minor = ["C", "c", "E", "e", "K", "k", "M", "N", "S", "s", "Fya", "Fyb", "Jka", "Jkb"]
        self.minor = ["C", "c", "E", "e", "K"][:SETTINGS.minor]
        # self.minor = SETTINGS.minor


        ##################
        # PATIENT GROUPS #
        ##################

        # Names of all considered patient groups (should correspond to keys of the dictionaries below).
        self.patgroups = ["ALA", "SCD", "Thal", "MDS", "AIHA", "Wu45", "Other"]

        # Distribution of patient groups in all hospital types.
        self.patgroup_distr = {
            "Other" : {"Other":1, "Wu45":0, "MDS":0, "Thal":0, "AIHA":0, "ALA":0, "SCD":0},
            "regional" : {"Other":0.886867, "Wu45":0.049352, "MDS":0, "Thal":0.008607969, "AIHA":0.014933, "ALA":0.031632, "SCD":0.008607969},
            "university" : {"Other":0.64605, "Wu45":0.10250, "MDS":0.04542, "Thal":0.05665, "AIHA":0.02731, "ALA":0.06543, "SCD":0.05665},
            "manual" : {"Other":0.64605, "Wu45":0.10250, "MDS":0.04542, "Thal":0.05665, "AIHA":0.02731, "ALA":0.06543, "SCD":0.05665},
        }

    
        ###########
        # WEIGHTS #
        ###########

        # Relative immunogenicity weights, to be used by the model as penalties for mismatching certain antigens.
        self.relimm_weights = pd.DataFrame(
            columns = ["A", "B", "D", "C",    "c",    "E",    "e",    "K",   "k", "M", "N", "S",    "s", "Fya",  "Fyb",  "Jka",  "Jkb"],
            data =   [[10,  10,  10,  0.0345, 0.0705, 0.2395, 0.0836, 0.3838, 0,  0,   0,   0.0131, 0,   0.0443, 0.0131, 0.0836, 0.0033]]
            )

        # Patient group specific weights, to be used by the model as penalties for mismatching certain antigens.
        self.patgroup_weights = pd.DataFrame(
            index =   ["ALA", "SCD", "Thal", "MDS", "AIHA", "Wu45", "Other"],
            columns = ["A", "B", "D", "C",    "c",    "E",    "e",    "K",   "k", "M", "N", "S",      "s",    "Fya",  "Fyb",  "Jka",  "Jkb" ],
            data =   [[10,  10,  10,  10,     10,     10,     10,     10,     0,  0,   0,   0.2020,   0.0769, 1.3635, 0.4040, 2.5756, 0.1010], # ALA
                      [10,  10,  10,  10,     10,     10,     10,     10,     0,  0,   0,   0.3367,   0.1282, 10,     0.6734, 10,     10    ], # SCD
                      [10,  10,  10,  10,     10,     10,     10,     10,     0,  0,   0,   0.1010,   0.0384, 0.6818, 0.2020, 1.2878, 0.0505], # Thal
                      [10,  10,  10,  10,     10,     10,     10,     10,     0,  0,   0,   0.1010,   0.0384, 0.6818, 0.2020, 1.2878, 0.0505], # MDS
                      [10,  10,  10,  10,     10,     10,     10,     10,     0,  0,   0,   0.2020,   0.0769, 1.3635, 0.4040, 2.5756, 0.1010], # AIHA
                      [10,  10,  10,  0.0265, 10,     10,     0.0644, 10,     0,  0,   0,   0.0034,   0.0013, 0.0227, 0.0067, 0.0429, 0.0017], # Wu45
                      [10,  10,  10,  0.0265, 0.0543, 0.1843, 0.0644, 0.2954, 0,  0,   0,   0.0034,   0.0013, 0.0227, 0.0067, 0.0429, 0.0017]] # Other
            )


        #####################
        # SUPPLY AND DEMAND #
        #####################

        # Each column specifies the probability of a request becoming known 0, 1, 2, etc. days before its issuing date.
        # CHANGE (no doubt)
        # self.request_lead_time_probabilities = {
        #     "Other" : [1/2, 1/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],               # Other = same day or day before
        #     "Wu45" : [1/2, 1/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],                # Wu45 = same day or day before
        #     "MDS" : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],                     # MDS = 7 days ahead
        #     "Thal" : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],                    # Thal = 7 days ahead
        #     "AIHA" : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],                    # AIHA = 7 days ahead
        #     "ALA" : [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 0, 0, 0, 0, 0, 0, 0],       # ALA = uniform between 0 and 6
        #     "SCD" : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]                      # SCD = 7 days ahead
        # }
        self.request_lead_time_probabilities = {
            "Other" : [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 0, 0, 0, 0, 0, 0, 0],     # Other = uniform between 0 and 6
            "Wu45" : [1/2, 1/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],                # Wu45 = 50/50 on the day or one day ahead.
            "MDS" : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],                     # MDS = 7 days ahead
            "Thal" : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],                    # Thal = 7 days ahead
            "AIHA" : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],                    # AIHA = on the same day
            "ALA" : [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 0, 0, 0, 0, 0, 0, 0],       # ALA = uniform between 0 and 6
            "SCD" : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]                      # SCD = 7 days ahead
        }

        # Each row specifies the probability of the corresponding patient type having a demand for [1,2,3,4] units respectively
        self.request_num_units_probabilities = {
            "Other" : [0.40437368293541415, 0.4968390449938975, 0.06828851313979055, 0.03049875893089782], 
            "Wu45" : [0.40437368293541415, 0.4968390449938975, 0.06828851313979055, 0.03049875893089782], 
            "MDS" : [0, 1, 0, 0], 
            "Thal" : [0, 1, 0, 0], 
            "AIHA" : [0, 1, 0, 0], 
            "ALA" : [0.40437368293541415, 0.4968390449938975, 0.06828851313979055, 0.03049875893089782], 
            "SCD" : [0, 1, 0, 0],
        }

        # Distribution of major blood groups in the donor population.
        self.donor_ABOD_distr = {"O-":0.1551, "O+":0.3731, "A-":0.0700, "A+":0.3074, "B-":0.0158, "B+":0.0604, "AB-":0.0047, "AB+":0.0134}

        # All possible antigen profiles for antigens A, B,
        # with 1 stating that the blood is positive for that antigen, and 0 that it is negative.
        self.ABO_phenotypes = [
            [ 0, 0 ],
            [ 0, 1 ],
            [ 1, 0 ],
            [ 1, 1 ]]

        # For each of the antigen profiles above, its prevalence in each of the ethnical populations.
        self.ABO_prevalences = {
            "Caucasian" :   [0.43, 0.09, 0.44, 0.04],
            "African" :     [0.27, 0.49, 0.2 , 0.04],
            "Asian" :       [0.27, 0.43, 0.25, 0.05]}

        # All possible antigen profiles for antigens D, C, c, E, e,
        # with 1 stating that the blood is positive for that antigen, and 0 that it is negative.
        self.Rhesus_phenotypes = [
            [0,1,1,1,1],
            [0,1,0,1,0],
            [0,1,0,0,1],
            [0,0,1,1,0],
            [0,0,1,0,1],
            [0,1,1,1,0],
            [0,1,1,0,1],
            [0,1,0,1,1],
            [0,0,1,1,1],
            [1,1,1,1,1],
            [1,1,0,1,0],
            [1,1,0,0,1],
            [1,0,1,1,0],
            [1,0,1,0,1],
            [1,1,1,1,0],
            [1,1,1,0,1],
            [1,1,0,1,1],
            [1,0,1,1,1]]

        # For each of the antigen profiles above, its prevalence in each of the ethnical populations.
        self.Rhesus_prevalences = {
            "Caucasian" :   [0.   , 0.   , 0.   , 0.   , 0.151, 0.   , 0.008, 0.   , 0.009, 0.133, 0.   , 0.185, 0.023, 0.021, 0.001, 0.349, 0.002, 0.118],
            "African" :     [0.   , 0.   , 0.   , 0.   , 0.068, 0.   , 0.   , 0.   , 0.   , 0.056, 0.   , 0.02 , 0.002, 0.458, 0.   , 0.21 , 0.   , 0.186],
            "Asian" :       [0.   , 0.   , 0.001, 0.001, 0.001, 0.   , 0.001, 0.   , 0.   , 0.303, 0.   , 0.518, 0.044, 0.003, 0.004, 0.085, 0.014, 0.025]}

        # All possible antigen profiles for antigens K, k,
        # with 1 stating that the blood is positive for that antigen, and 0 that it is negative.
        self.Kell_phenotypes = [
            [ 0, 0 ],
            [ 1, 0 ],
            [ 0, 1 ],
            [ 1, 1 ]]

        # For each of the antigen profiles above, its prevalence in each of the ethnical populations.
        self.Kell_prevalences = {
            "Caucasian" :   [0.   , 0.002, 0.91 , 0.088],
            "African" :     [0.   , 0.   , 0.98 , 0.02 ],
            "Asian" :       [0.   , 0.   , 1.   , 0.   ]}

        # All possible antigen profiles for antigens M, N, S, s,
        # with 1 stating that the blood is positive for that antigen, and 0 that it is negative.
        self.MNS_phenotypes = [
            [1,0,1,0],
            [1,0,1,1],
            [1,0,0,1],
            [1,1,1,0],
            [1,1,1,1],
            [1,1,0,1],
            [0,1,1,0],
            [0,1,1,1],
            [0,1,0,1],
            [1,0,0,0],
            [1,1,0,0],
            [0,1,0,0]]

        # For each of the antigen profiles above, its prevalence in each of the ethnical populations.
        self.MNS_prevalences  = {
            "Caucasian" :   [0.06 , 0.14 , 0.08 , 0.04 , 0.24 , 0.22 , 0.01 , 0.06 , 0.15 , 0.   , 0.   , 0.   ],
            "African" :     [0.02 , 0.07 , 0.16 , 0.02 , 0.13 , 0.325, 0.02 , 0.05 , 0.19 , 0.004, 0.004, 0.007],
            "Asian" :       [0.06 , 0.14 , 0.08 , 0.04 , 0.24 , 0.22 , 0.01 , 0.06 , 0.15 , 0.   , 0.   , 0.   ]}

        # All possible antigen profiles for antigens Fya, Fyb, 
        # with 1 stating that the blood is positive for that antigen, and 0 that it is negative.
        self.Duffy_phenotypes = [
            [ 0, 0 ],
            [ 1, 0 ],
            [ 0, 1 ],
            [ 1, 1 ]]

        # For each of the antigen profiles above, its prevalence in each of the ethnical populations.
        self.Duffy_prevalences = {
            "Caucasian" :   [0.   , 0.17 , 0.34 , 0.49 ],
            "African" :     [0.68 , 0.09 , 0.22 , 0.01 ],
            "Asian" :       [0.   , 0.908, 0.003, 0.089]}

        # All possible antigen profiles for antigens Jka, Jkb,
        # with 1 stating that the blood is positive for that antigen, and 0 that it is negative.
        self.Kidd_phenotypes = [
            [ 0, 0 ],
            [ 1, 0 ],
            [ 0, 1 ],
            [ 1, 1 ]]

        # For each of the antigen profiles above, its prevalence in each of the ethnical populations.
        self.Kidd_prevalences = {
            "Caucasian" :   [0.   , 0.263, 0.234, 0.503],
            "African" :     [0.   , 0.511, 0.081, 0.488],
            "Asian" :       [0.009, 0.232, 0.268, 0.491]}

        
        ##########
        # GUROBI #
        ##########
        
        self.status_code = {
            1 : "MODEL IS LOADED, BUT NO SOLUTION IS AVAILABLE",
            2 : "MODEL IS SOLVED OPTIMALLY",
            3 : "MODEL IS INFEASIBLE",
            4 : "MODEL IS EITHER INFEASIBLE OR UNBOUNDED\nTo obtain a more definitive conclusion, set the DualReductions parameter to 0 and reoptimize.",
            5 : "MODEL IS UNBOUNDED\nAn unbounded ray allows the objective to improve without limit.",
            6 : "NO SOLUTION AVAILABLE\nThe optimal objective was worse than the Cutoff parameter.",
            7 : "OPTIMIZATION TERMINATED\nIterationLimit or BarIterLimit parameter was exceeded.",
            8 : "OPTIMIZATION TERMINATED\nNodeLimit parameter was exceeded.",
            9 : "OPTIMIZATION TERMINATED\nTimeLimit parameter was exceeded.",
            10 : "OPTIMIZATION TERMINATED\nSolutionLimit parameter was exceeded.",
            11 : "OPTIMIZATION TERMINATED BY USER\nObtained results are not saved.",
            12 : "OPTIMIZATION TERMINATED\nUnrecoverable numerical difficulties.",
            13 : "UNABLE TO SATISFY OPTIMALITY\nA sub-optimal solution is available.",
            14 : "ASYNCHRONOUS CALL WAS MADE, ASSOCIATED OPTIMIZATION NOT YET COMPLETE",
            15 : "LIMIT SET BY USER WAS EXCEEDED\nThis is either a bound on the best objective or the best bound."
        }
