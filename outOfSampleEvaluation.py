
from SingleRegionModel_fixedFirstStage import SingleRegionModel_fixedFirstStage
import numpy as np
import pickle
from pyomo.opt import SolverFactory
from pyomo.environ import *
from datetime import datetime
import pandas as pd
from pathlib import Path
import time
import sys
from gurobipy import GRB
from datetime import datetime

def evaluate_oos_objective_function(first_stage_solution, scenario_set):
    
    opt = SolverFactory("gurobi_direct")
    opt.options['Method'] = 2
    
    objs = []
    
    for n in range(scenario_set.shape[0]):
    
        print(f'scenario {n} running...', flush=True)

        fixed_surrogate_instance = SingleRegionModel_fixedFirstStage(short_term_scenarios=scenario_set[n], num_short_term_scenarios=1, first_stage_solution=first_stage_solution)
        fixed_surrogate = fixed_surrogate_instance.get_model()
        results = opt.solve(fixed_surrogate)
        objs.append(float(value(fixed_surrogate.objective)))

    return objs

print(f'running start {datetime.now()}', flush=True)

test_scenarios = np.load('test_scenarios.npy')
scenario_set = test_scenarios[np.random.choice(test_scenarios.shape[0], size=200)]

sol_folder = Path('surrogate_sol')

print('calculating the oos for the deterministic equivalent', flush=True)
with open(f'sol_det.pkl','rb') as f:
            first_stage_sol_deterministic = pickle.load(f)
            
deterministic_objs = evaluate_oos_objective_function(first_stage_sol_deterministic, scenario_set)

solutions_df  = pd.DataFrame({'deterministic':deterministic_objs})
i = 0
for file in sol_folder.glob('*.pkl'):
    
    print(f'calculating oos evaluation for {file.name}', flush=True)
    
    with open(file,'rb') as f:
            first_stage_sol_surrogate = pickle.load(f)

    surrogate_objs = evaluate_oos_objective_function(first_stage_sol_surrogate, scenario_set)
    
    print(f'Printing solution for the surrogate model: ', flush=True)
    print(surrogate_objs, flush=True)
    print(f'Printing solution for the deterministic model: ', flush=True)
    print(deterministic_objs, flush=True)

    solutions_df[f'surrogate_{i}'] = surrogate_objs
    i += 1

solutions_df.to_csv(f'oos_solutions.csv')

print(f'run ended {datetime.now()}', flush=True)