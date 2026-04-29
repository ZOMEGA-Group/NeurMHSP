from SingleRegionModel_NN import SingleRegionProblem_NN
from tensorflow import keras
import pickle
from new_formulation.SingleRegionModel import SingleRegionModelNew
from pyomo.opt import SolverFactory
from pyomo.environ import *
import re
from pathlib import Path
import time
import sys
import tensorflow as tf
from gurobipy import GRB                              
from datetime import datetime
import pandas as pd
import numpy as np
import os
import ast
import argparse

def log_mae_loss(y_true, y_pred):
    epsilon = 1e-6
    y_true_safe = tf.clip_by_value(y_true, 0.0, 1e6)
    y_pred_safe = tf.clip_by_value(y_pred, 0.0, 1e6)
    
    y_true_log = tf.math.log1p(y_true_safe + epsilon)
    y_pred_log = tf.math.log1p(y_pred_safe + epsilon)
    return tf.reduce_mean(tf.abs(y_true_log - y_pred_log))

def save_first_stage_surrogate(model):
    all_vars = model.getVars()
    variables = model.getAttr('X', all_vars)
    names = model.getAttr('VarName', all_vars)

    surrogate_x_acc = {}
    surrogate_x_inst = {}
    for index in range(len(names)):
        
        if 'x_acc' in names[index]:
            match = re.findall(r'\d+', names[index])
            i, j = map(int, match)
            surrogate_x_acc[(i, j)] = variables[index]
        
        if 'x_inst' in names[index]:
            match = re.findall(r'\d+', names[index])
            i, j = map(int, match)
            surrogate_x_inst[(i, j)] = variables[index]
    
    return [surrogate_x_acc, surrogate_x_inst]


    
def save_first_stage_deterministic(model):
    deterministic_x_inst = {}
    for (p,i), var in model.x_inst.items():
        deterministic_x_inst[p,i] = value(var)

    deterministic_x_acc = {}
    for (p,i), var in model.x_acc.items():
        deterministic_x_acc[p,i] = value(var)

    return [deterministic_x_acc, deterministic_x_inst]

def get_solution_vector(solution_dict):
    rows = max(i for i, j in solution_dict.keys()) + 1
    cols = max(j for i, j in solution_dict.keys()) + 1
    array = np.zeros((rows, cols))
    
    for (i, j), value in solution_dict.items():
        array[i, j] = value
    
    return array


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stability test and comparison between surrogate and DE')
    parser.add_argument('--num_scenarios', type=int, required=True, help='number of short term scenarios')
    
    
    args = parser.parse_args()

    with open(f"results/deterministic_surrogate_comparison_independent_5scenario.txt", "w") as file:
        sys.stdout = file
        
        df_scen = pd.read_csv('scenarios/scenario_dataset_cumulate.csv')
        
        results_df = {}
        
        n_experiments = 5
        num_scenarios = args.num_scenarios
        
        deterministic_objs = []
        deterministic_times = []
        for n in range(n_experiments):
            
            print(f'solving the deterministic equivalent {n}')
            
            scenarios = np.load('scenarios/scenarios.npy')
            df_scen = df_scen[df_scen['scenario_number'] == num_scenarios]
            indexes = df_scen[df_scen['dataset_id'] == n]['indexes'].iloc[0]
            index_list = ast.literal_eval(indexes)
            scenarios = scenarios[index_list]
            
            model_deterministic_instance = SingleRegionModelNew(short_term_scenarios=scenarios, num_short_term_scenarios=args.num_scenarios)
            model_deterministic = model_deterministic_instance.get_model()
            opt = SolverFactory("gurobi_direct")
            opt.options['Method'] = 2

            start_time = time.time()
            results = opt.solve(model_deterministic, tee=True)
            end_time = time.time()
            solving_time = end_time - start_time
            print(f'deterministic model time: {solving_time}')
            deterministic_times.append(solving_time)
            deterministic_equivalent = float(value(model_deterministic.objective))
            deterministic_objs.append(deterministic_equivalent)
            print(deterministic_equivalent)
            
            first_stage_sol_deterministic = save_first_stage_deterministic(model_deterministic)
                    
            with open(f'first_stage_sol/sol_det_{n}_5s_independent.pkl','wb') as f:
                    pickle.dump(first_stage_sol_deterministic, f)
        
        results_df['deterministic_objs'] = deterministic_objs
        results_df['deterministic_time'] = deterministic_times
        results_df = pd.DataFrame(results_df)
        results_df.to_csv('in_sample_results_independent_5scen.csv')   
        
        n_models = 5
        
        for n in range(n_models):
            models_folder = Path(f'NN_models/NN_models_5scenario_{n}')

            surrogate_objs = []
            surrogate_times = []
            surrogate_deltas = []
            inst_dinstances = []
            acc_dinstances = []
            
            for exp in range(5):
                
                print(exp)
                filename = f'NN_model_5_scenarios_{n}_{exp}.keras'
                model_path = os.path.join(models_folder, filename)
                loaded_model = keras.models.load_model(model_path, custom_objects={'log_mae_loss': log_mae_loss})
                with open(f'Scalers/scaler_5_scenario_{exp}.pkl','rb') as f:
                    loaded_scaler = pickle.load(f)
                
                
                print(f'loaded model: {model_path}')
                print('solving the surrogate model...')
                model_surrogate_instance = SingleRegionProblem_NN(nn_model= loaded_model, scaler=loaded_scaler)
                model_surrogate = model_surrogate_instance.get_model()
                model_surrogate.setParam("Method", 2)
                model_surrogate.setParam("TimeLimit", 10800)
                start_time = time.time()
                model_surrogate.optimize()
                end_time = time.time()
                solving_time = end_time - start_time
                print(f'surrogate model time: {solving_time}')
                surrogate_times.append(solving_time)
                status = model_surrogate.Status
                
                if status == GRB.OPTIMAL or status == GRB.TIME_LIMIT or status == GRB.SUBOPTIMAL:
                    if status == GRB.OPTIMAL or status == GRB.SUBOPTIMAL:
                        nn_embedded = model_surrogate.ObjVal  
                        surrogate_objs.append(nn_embedded)
                        
                        first_stage_surrogate = save_first_stage_surrogate(model_surrogate)

                        dist_inst = np.linalg.norm(get_solution_vector(first_stage_surrogate[1]) - get_solution_vector(first_stage_sol_deterministic[1]))
                        dist_acc = np.linalg.norm(get_solution_vector(first_stage_surrogate[0]) - get_solution_vector(first_stage_sol_deterministic[0]))
                        print('Distance between X_inst: ', dist_inst)
                        print('Distance between X_acc: ', dist_acc)
                        
                        inst_dinstances.append(dist_inst)
                        acc_dinstances.append(dist_acc)
                        
                        with open(f'first_stage_sol/sol_sur_{n}_{exp}_5s_independent.pkl','wb') as f:
                            pickle.dump(first_stage_surrogate, f)
                        
                        print(nn_embedded)
                
                        det_eq = results_df.iloc[exp]['deterministic_objs']
                        delta = float(((det_eq - nn_embedded) / det_eq))
                        print('delta: ',delta )
                        surrogate_deltas.append(delta)
                        
                    elif status == GRB.TIME_LIMIT:
                        nn_embedded = model_surrogate.ObjBound
                        surrogate_objs.append(nn_embedded)
                        inst_dinstances.append(-1)
                        acc_dinstances.append(-1)
                        det_eq = results_df.iloc[exp]['deterministic_objs']
                        delta = float(((det_eq - nn_embedded) / det_eq))
                        print('delta: ',delta )
                        surrogate_deltas.append(delta)
                        
                        
                        if exp != 4:
                            for tmp in range(exp+1, n_experiments):
                                surrogate_objs.append(-1)
                                inst_dinstances.append(-1)
                                acc_dinstances.append(-1)
                                surrogate_deltas.append(-1)
                                surrogate_times.append(-1)
                            
                        break
                else:
                    print("model not solved correctly", status)
                    surrogate_objs.append(-1)
                    inst_dinstances.append(-1)
                    acc_dinstances.append(-1)
                    surrogate_deltas.append(-1)
            
            
            results_df[f'Surrogate_{n}_objs'] = surrogate_objs
            results_df[f'Surrogate_{n}_times'] = surrogate_times
            results_df[f'Surrogate_{n}_deltas'] = surrogate_deltas
            results_df[f'Surrogate_{n}_inst'] = inst_dinstances
            results_df[f'Surrogate_{n}_acc'] = acc_dinstances
            results_df.to_csv('in_sample_results_independent_5scen.csv')   
        
        
        results_df.to_csv('in_sample_results_independent_5scen.csv')        
    
    
