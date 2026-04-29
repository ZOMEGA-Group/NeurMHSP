import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import random

def discretize_bounds(row):
    return np.linspace(row['lower'], row['upper'], 50)

def generate_random_solutions(num_solutions):
    
    node_to_stage = {
        0: 1,
        1: 1,
        2: 1,
        3: 2,
        4: 2,
        5: 2,
        6: 2,
        7: 2,
        8: 2,
        9: 2,
        10: 2,
        11: 2
    }
    
    investment_data = pd.read_csv('cleaned/single_investment_data.csv')
    
    operational_nodes = range(12)
    technologies = range(11)

    lower_bounds = []

    for i in operational_nodes:
        for tech in technologies:
            row = investment_data.loc[tech]
            
            if i < 3:
                lower = row['capacity_at_5']
            else:
                lower = row['capacity_at_10']
                
            lower_bounds.append({'ope_node': i, 'technology': tech, 'lower': lower})

    df_lower = pd.DataFrame(lower_bounds)
    X_acc_bounds = df_lower.pivot(index='ope_node', columns='technology', values='lower')
    X_acc_bounds = X_acc_bounds.reset_index()
    
    X_inst_bounds = pd.DataFrame(columns = ['technology', 'ope_node', 'upper', 'lower'])

    for i in operational_nodes:
        for p in technologies:
            
            if i in [0,1,2]:
                
                if p == 5:
                    
                    if 15 + investment_data.loc[p]['capacity_at_5'] > investment_data.loc[p]['Max_capacity (GW)']:
                        upper = 15 - investment_data.loc[p]['capacity_at_5']
                    else:
                        upper = 15 
                else:
                    
                    if 77.5 + investment_data.loc[p]['capacity_at_5'] > investment_data.loc[p]['Max_capacity (GW)']:
                        upper = 77.5 - investment_data.loc[p]['capacity_at_5']
                    else:
                        upper = 77.5 
            else:
                if p == 5:
                    if 15 + investment_data.loc[p]['capacity_at_10'] > investment_data.loc[p]['Max_capacity (GW)']:
                        upper = 15 - investment_data.loc[p]['capacity_at_10']
                    else:
                        upper = 15  
                else:
                    if 155 + investment_data.loc[p]['capacity_at_10'] > investment_data.loc[p]['Max_capacity (GW)']:
                        upper = 155 - investment_data.loc[p]['capacity_at_10']
                    else:
                        upper = 155  
            
            df_row = pd.DataFrame({'technology': p, 'ope_node': i, 'upper':upper, 'lower':0}, index=[0])
            X_inst_bounds = pd.concat([X_inst_bounds, df_row])
            
    X_inst_bounds['stage'] = X_inst_bounds['ope_node'].map(node_to_stage)
    X_inst_bounds = X_inst_bounds.reset_index()
    
    df_discretized = X_inst_bounds.apply(discretize_bounds, axis=1, result_type='expand')
    df_discretized.columns = [f'point_{i+1}' for i in range(df_discretized.shape[1])]
    df_discretized = df_discretized.reset_index().drop('index', axis=1)
    X_inst = pd.concat([X_inst_bounds, df_discretized], axis = 1)
    
    max_investments_discretizetion = [5,10,15,20,25,30,35,40,45,50]
    
    results = []
    sample_sums = []
    for _ in range(num_solutions):
        rand_node = np.random.choice(operational_nodes)
        node_data = X_inst[X_inst['ope_node'] == rand_node].reset_index(drop=True)
        
        rand_index = np.random.choice(max_investments_discretizetion)
        max_inv = node_data[f'point_{str(rand_index)}'].iloc[0]
        
        stage = node_data['stage'].iloc[0]
        max_total = max_inv
        

        sample_sum = 0.0
        samples = [0.0] * len(node_data)
        
        tech_indices = list(node_data.index)
        np.random.shuffle(tech_indices)
        

        for idx in tech_indices:
            row = node_data.loc[idx]
            values = row.iloc[6:56].values.astype(float)
            num_iter = 0
            
            while True:
                val = random.choices(values)[0]
                num_iter += 1
                if sample_sum + val <= max_total:
                    samples[idx] = val
                    sample_sum += val
                    break
                elif num_iter > 150:
                    
                    samples[idx] = 0
                    num_iter = 0
        sample_sums.append(sample_sum)
        dataset_row = {'node': rand_node}
        for i, val in enumerate(samples):
            dataset_row[i] = val
        dataset_row['ope_cost'] = None
        results.append(dataset_row)

    generated_df = pd.DataFrame(results)
    
    other_params = pd.read_csv('cleaned/other_params.csv')
    demand_scaling = pd.read_csv('cleaned/demand_scaling_factor.csv')
    co2_scaling = pd.read_csv('cleaned/co2_budget_scaling_factor.csv')
    generated_df['co2_budget'] =   ((generated_df['node'] + 1).map(co2_scaling.set_index('node')['value']) * other_params['co2_lim (ktCO2)'].iloc[0])
    generated_df['co2_price'] = other_params['co2_cost (mn£/ktCO2)'].iloc[0]
    generated_df['demand_scaling'] = ((generated_df['node'] + 1).map(demand_scaling.set_index('node')['value']))
    
    merged = generated_df.merge(X_acc_bounds, right_on='ope_node', left_on='node', suffixes=('', '_other'))
    tech_cols = list(range(11))

    for col in tech_cols:
        merged[str(col)] = merged[str(col)] + merged[f"{col}_other"]
    generated_df = merged.drop(columns=[f"{col}_other" for col in tech_cols])
    
    return generated_df, X_acc_bounds


def add_base_values(X_acc, generated_df):
    
    operational_nodes = range(12)
    
    base = []
    for i in operational_nodes:
        ope_node = X_acc[X_acc['ope_node'] == i].reset_index().drop('index', axis = 1)

        dataset_row = {'node':i}
        for col in [0,1,2,3,4,5,6,7,8,9,10]:
            var = ope_node[col].iloc[0]
            dataset_row[str(col)] = var
            
        base.append(dataset_row)
        
    base_df = pd.DataFrame(base)
    base_df['ope_cost'] = None
    
    other_params = pd.read_csv('cleaned/other_params.csv')
    demand_scaling = pd.read_csv('cleaned/demand_scaling_factor.csv')
    co2_scaling = pd.read_csv('cleaned/co2_budget_scaling_factor.csv')
    base_df['co2_budget'] =   ((base_df['node'] + 1).map(co2_scaling.set_index('node')['value']) * other_params['co2_lim (ktCO2)'].iloc[0])
    base_df['co2_price'] = other_params['co2_cost (mn£/ktCO2)'].iloc[0]
    base_df['demand_scaling'] = ((base_df['node'] + 1).map(demand_scaling.set_index('node')['value']))
    
    return pd.concat([generated_df, base_df], ignore_index = True)


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset builder for SRP model.')
    parser.add_argument('--num_solutions', type=int, required=True, help='number of random solutions')
    
    args = parser.parse_args()

    print(f'running start {datetime.now()}')

    print('generating random solutions...')
    generated_df, X_acc = generate_random_solutions(args.num_solutions)


    print('adding base values...')
    generated_df = add_base_values(X_acc, generated_df)

    generated_df.to_csv('Random solutions/random_solutions_50_2.csv')
    
    print(f'running end {datetime.now()}')