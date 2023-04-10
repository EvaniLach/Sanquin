from gurobipy import *
import numpy as np

from blood import *
# from log import *

# Single-hospital setup: MINRAR model for matching within a single hospital.
# I and R are both lists of blood groups (int format), where I = issued products, R = today's requests
def action_to_matches(PARAMS, I, R, day, df):

    ################
    ## PARAMETERS ##
    ################

    antigens = PARAMS.major + PARAMS.minor
    A = {antigens[k] : k for k in range(len(antigens))}
    A_no_Fyb = [A[ag] for ag in A.keys() if ag != "Fyb"]

    # Retrieve the antigen (and patient group) weights.
    w = np.array(PARAMS.relimm_weights[antigens])[0]

    _I = list(range(len(I)))
    _R = list(range(len(R)))

    # I x R x A, for each combination for i and r, whether the match is compatible on a (1) or not (0)
    C = precompute_compatibility(I, R)

    ############
    ## GUROBI ##
    ############

    model = Model(name="model")
    model.Params.LogToConsole = 0

    ###############
    ## VARIABLES ##
    ###############

    # x: For each request r∈R and inventory product i∈I, x[i,r] = 1 if r is satisfied by i, 0 otherwise.
    x = model.addVars(len(I), len(R), name='x', vtype=GRB.BINARY, lb=0, ub=1)

    model.update()
    model.ModelSense = GRB.MINIMIZE

    # Remove variable x[i,r] if the match is not compatible on the major antigens.
    for i in _I:
        for r in _R:
            # if (comp_antigens(C[i,r], 3) != 0):
            if sum(C[i,r][:3]) != 0:
                model.remove(x[i,r])

    #################
    ## CONSTRAINTS ##
    #################

    # For each inventory product i∈I, ensure that i can not be issued more than once, and each request r can receive a maximum of one product.
    model.addConstrs(quicksum(x[i,r] for r in _R) <= 1 for i in _I)
    model.addConstrs(quicksum(x[i,r] for i in _I) <= 1 for r in _R)

    ################
    ## OBJECTIVES ##
    ################

    model.setObjectiveN(expr =  quicksum(1 - quicksum(x[i,r] for i in _I) for r in _R), index=0, priority=1, name="shortages")
    if "Fyb" in antigens:
        model.setObjectiveN(expr =  quicksum(
                                        quicksum(x[i,r] * C[i,r,k] * w[k] for k in A_no_Fyb)
                                        + x[i,r] * C[i,r,A["Fyb"]] * int(bin(r)[A["Fya"]+2]) * w[A["Fyb"]]
                                    for i in _I for r in _R), index=1, priority=0, name="mismatches")
    else:
        model.setObjectiveN(expr =  quicksum(
                                        quicksum(x[i,r] * C[i,r,k] * w[k] for k in A_no_Fyb)
                                    for i in _I for r in _R), index=1, priority=0, name="mismatches")

    model.optimize()

    shortages = model.getObjective(0).getValue()
    mismatches = model.getObjective(1).getValue()

    x = np.zeros([len(I), len(R)])

    for var in model.getVars():
        var_name = re.split(r'\W+', var.varName)[0]
        if var_name == "x":
            index0 = int(re.split(r'\W+', var.varName)[1])
            index1 = int(re.split(r'\W+', var.varName)[2])
            x[index0, index1] = var.X

    xi = x.sum(axis=1)
    assigned = [i for i in _I if xi[i] > 0]
    discarded = [I[i] for i in _I if xi[i] == 0]

    for i in assigned:
        for r in _R:
            for ag in [ag for ag in A.keys() if C[i,r,A[ag]] == 0]:
                if (ag != "Fyb") or (bin(r)[A["Fya"]+2] == 1):
                    df.loc[day,[f"num mismatches {ag}"]] += 1

    return shortages, mismatches, [I[i] for i in assigned], discarded, df