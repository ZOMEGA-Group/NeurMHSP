import pandas as pd
import numpy as np
import argparse
from pyomo.environ import *
import sys
import os
from datetime import datetime
import ast


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from old_formulation.OperationalSubProblem_independentScenario import OperationalSubProblem


def calculate_operational_cost(generated_df, num_scenarios, first_scenario):
    
    scenarios = np.arange(first_scenario, first_scenario + num_scenarios)
    generated_df = generated_df.loc[generated_df.index.repeat(num_scenarios)].reset_index(drop=True)
    generated_df["scenario"] = np.tile(scenarios, len(generated_df) // num_scenarios)
    generated_df["ope_cost"] = None
    scenario_set = np.load('scenarios/scenarios.npy')
    
    P_accg = generated_df[['0','1','2','3','4','5']]
    P_accSE = generated_df[['6','7']]
    P_accr = generated_df[['8','9','10']]
    
    for index, row in generated_df.iterrows():
        
        model_instance = OperationalSubProblem(scenario_set[row['scenario']], num_short_term_scenario=1)
        model = model_instance.get_model()
        opt = SolverFactory("gurobi_direct")
        opt.options['Method'] = 2
        
        
        for g in model.G:
            model.p_accG[g].fix(P_accg[str(g)].iloc[index])
            for t in model.T:
                model.p_G[g,t].setub(P_accg[str(g)].iloc[index])
        
        for s in model.S:
            model.p_accSE[s].fix(P_accSE[str(s+6)].iloc[index])
            for t in model.T:
                model.p_SEm[s,t].setub(P_accSE[str(s+6)].iloc[index])
                model.p_SEp[s,t].setub(P_accSE[str(s+6)].iloc[index])
                model.q_SE[s,t].setub(P_accSE[str(s+6)].iloc[index] * model.gamma_se[s])
        
          

        
        for r in model.R:
            model.p_accR[r].fix(P_accr[str(r+8)].iloc[index])
        
        model.mu_dp.set_value(generated_df['demand_scaling'].iloc[index])
        model.mu_e.set_value((generated_df['co2_budget'].iloc[index]))
    
        result = opt.solve(model)
        
        
        if result.solver.termination_condition == TerminationCondition.infeasible:
            print('infeasible at index', index)
            generated_df.at[index, 'ope_cost'] = -1
        else:
            ope_value = float(value(model.objective))
            generated_df.at[index, 'ope_cost'] = ope_value

        for var in model.component_data_objects(Var, active=True):
            var.unfix()
            var.value = None
    
    return generated_df


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset builder for SRP model.')
    parser.add_argument('--num_scenarios', type=int, required=True, help='number of short term scenarios')
    parser.add_argument('--first_scenario', type=int, required=True, help='first index of short term scenarios')
    parser.add_argument('--dataset_name', type=str, required=True, help='name of the saved dataset')
    
    args = parser.parse_args()

    df = pd.read_csv('Random Solutions/random_solutions_2.csv')

    print(f'calculating operational costs {datetime.now()}...')
    df = calculate_operational_cost(df, args.num_scenarios, args.first_scenario)

    df.to_csv(f'{args.dataset_name}.csv')

    print(f'running end {datetime.now()}')