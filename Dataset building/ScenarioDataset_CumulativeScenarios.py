import pandas as pd
import numpy as np
import os
import argparse


def categorize_wind_scenarios(scenario_set):
   
    wind = scenario_set[:, 0, :]              
    wind_means = wind.mean(axis=1)           
    
    low_thr, high_thr = np.quantile(wind_means, [1/3, 2/3])
    
    low_idx = np.where(wind_means <= low_thr)[0]
    med_idx = np.where((wind_means > low_thr) & (wind_means <= high_thr))[0]
    high_idx = np.where(wind_means > high_thr)[0]
    
    return {
        'low': list(low_idx),
        'medium': list(med_idx),
        'high': list(high_idx)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset builder for SRP model.')
    parser.add_argument('--num_scenarios', type=int, required=True, help='number of scenarios')
    parser.add_argument('--num_datasets', type=int, required=True, help='number of datasets')
    args = parser.parse_args()

    file_path = 'scenarios/scenario_dataset_cumulate.csv'

    if os.path.exists(file_path):
        scen_df = pd.read_csv(file_path)
    else:
        scen_df = pd.DataFrame(columns = ['scenario_number', 'dataset_id', 'indexes'])
        
    scenario_set = np.load('scenarios/scenarios.npy')
    scenario_set = scenario_set[:200]
    wind_groups = categorize_wind_scenarios(scenario_set)
    total_available = range(200)
    existing_sizes = scen_df['scenario_number'].unique() if not scen_df.empty else []
    max_existing = max(existing_sizes) if len(existing_sizes) > 0 else 0
    
    scen_df_tmp = scen_df[scen_df['scenario_number'] == max_existing]
    
    if args.num_scenarios > max_existing:
        rows = []
        for d in range(args.num_datasets):
            if len(scen_df_tmp) > 0:
                base_set = scen_df_tmp[scen_df_tmp['dataset_id'] == d]['indexes'].iloc[0]
                base_set = [int(x) for x in base_set.strip("[]").split(",")] if isinstance(base_set, str) else list(base_set)
                
                needed = args.num_scenarios - max_existing
                available = list(set(total_available) - set(base_set))
            else:
                base_set = []
                available = list(total_available)
                needed = args.num_scenarios
            
            
            available_low = list(set(wind_groups['low']) & set(available))
            available_med = list(set(wind_groups['medium']) & set(available))
            available_high = list(set(wind_groups['high']) & set(available))
            
            n_each = [needed // 3, needed // 3, needed - 2 * (needed // 3)]

            sampled = []
            sampled += list(int(x) for x in np.random.choice(available_low, size=min(len(available_low), n_each[0]), replace=False))
            sampled += list(int(x) for x in np.random.choice(available_med, size=min(len(available_med), n_each[1]), replace=False))
            sampled += list(int(x) for x in np.random.choice(available_high, size=min(len(available_high), n_each[2]), replace=False))

            
            if len(sampled) < needed:
                remaining_available = list(set(available) - set(sampled))
                extra = list(int(x) for x in np.random.choice(remaining_available, size=needed - len(sampled), replace=False))
                sampled += list(extra)

            final_set = list(int(x) for x in (base_set + sampled))
            row = {'scenario_number': args.num_scenarios, 'dataset_id': d, 'indexes': final_set}
            rows.append(row)
    
                                                             
    df_tmp = pd.DataFrame(rows)
    df = pd.concat([scen_df, df_tmp], ignore_index=True)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    
    df.to_csv(file_path)
    
    
    
   


