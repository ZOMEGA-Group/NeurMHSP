from pyomo.environ import *
import pandas as pd
import numpy as np

class SingleRegionModelNew:
    
    def __init__(self, short_term_scenarios, num_short_term_scenarios):
        self.model = ConcreteModel()
        self.short_term_scenarios = short_term_scenarios
        self.num_short_term_scenario = num_short_term_scenarios
        self.build_model()
        

    def build_model(self):
        
        inv_data = pd.read_csv('cleaned/single_investment_data.csv')
        conventional_param = pd.read_csv('cleaned/conventional_params.csv')
        storage_param = pd.read_csv('cleaned/storage_params.csv')
        other_params = pd.read_csv('cleaned/other_params.csv')
        demand_scaling = pd.read_csv('cleaned/demand_scaling_factor.csv')
        co2_scaling = pd.read_csv('cleaned/co2_budget_scaling_factor.csv')
        
        scenarios_infos = pd.read_csv('scenarios/scenarios_info.csv')
        
        
        
        scenarios = self.short_term_scenarios

        # investment planning self.model sets
        self.model.P = Set(initialize=range(11))
        self.model.I_0 = Set(initialize=range(13))
        self.model.I = Set(initialize=range(12))
        
        ti_set = {
            0: [0],
            1: [0],
            2: [0],
            3: [1, 0],
            4: [1, 0],
            5: [1, 0],
            6: [2, 0],
            7: [2, 0],
            8: [2, 0],
            9: [3, 0],
            10: [3, 0],
            11: [3, 0]
        } 
            
        
        self.model.I_i = Set(self.model.I, initialize=ti_set) 

        # operational self.model sets
        self.model.N = Set(initialize=range(self.num_short_term_scenario))
        self.model.T = Set(initialize=range(int(scenarios_infos['scenario_length'].iloc[0]) * self.num_short_term_scenario))
        self.model.G = Set(initialize=range(6))
        self.model.S = Set(initialize=range(2))
        self.model.R = Set(initialize=range(3))
        
        tn_set = {}
        for n in self.model.N:
            
            start = n * int(scenarios_infos['scenario_length'].iloc[0])
            end = start + int(scenarios_infos['scenario_length'].iloc[0])
            tn_set[n] = range(start, end)
            
        
        
        self.model.T_n = Set(self.model.N, initialize=tn_set)
        
        self.model.P_to_G = Set(initialize=[0,1,2,3,4,5])
        self.model.P_to_S = Set(initialize=[6,7])
        self.model.P_to_R = Set(initialize=[8,9,10])

        # investment planning model parameters
        self.model.C_inv = Param(self.model.P, self.model.I_0, initialize = {(index, i): value for index, value in enumerate(inv_data['CapeX (mn£/GW)']) for i in self.model.I_0})
        self.model.C_fix = Param(self.model.P, self.model.I, initialize= {(index, i): value for index, value in enumerate(inv_data['FixOM (mn£/GWyr)']) for i in self.model.I})
        
        xinit = {}
        for p in self.model.P:
            for i in self.model.I:
        
                if i < 3:
                    xinit[(p,i)] = float(inv_data['capacity_at_5'][p])
                else:
                    xinit[(p,i)] = float(inv_data['capacity_at_10'][p])
                    
        self.model.X_init = Param(self.model.P, self.model.I, initialize = xinit)
        self.model.X_max = Param(self.model.P, initialize = {index:value for index, value in enumerate(inv_data['Max_capacity (GW)'])})
        self.model.k = Param(initialize = 5)
        
        def calculate_discount_factor(period):
            return 1 / ((1+0.05) ** period)
        
        delta_i = {}
        for i in self.model.I:
            if i < 3:
                delta_i[i] = calculate_discount_factor(5)
            else:
                delta_i[i] = calculate_discount_factor(10)
                
        delta_i_0 = {}
        for i in self.model.I_0:
            if i == 0:
                delta_i_0[i] = calculate_discount_factor(0)
            elif i < 4:
                delta_i_0[i] = calculate_discount_factor(5)
            else:
                delta_i_0[i] = calculate_discount_factor(10)
        
        self.model.delta_I_0 = Param(self.model.I_0, initialize = delta_i_0)
        self.model.delta_I = Param(self.model.I, initialize = delta_i)
        
        prob = 1/self.num_short_term_scenario
        self.model.pi_n = Param(self.model.N, initialize = {index:prob for index, _ in enumerate(self.model.N)})
        
        pi_i_0 = {
            0:1,
            1:0.33,
            2:0.33,
            3:0.33,
            4:0.11,
            5:0.11,
            6:0.11,
            7:0.11,
            8:0.11,
            9:0.11,
            10:0.11,
            11:0.11,
            12:0.11
        }
        
        
        pi_i = {}
        for i in self.model.I:
            if i < 3:
                pi_i[i] = 1/3
            else:
                pi_i[i] = 1/9
        
        self.model.pi_I_0 = Param(self.model.I_0, initialize = pi_i_0)
        self.model.pi_I = Param(self.model.I, initialize = pi_i)
        self.model.H_p = Param(self.model.P, initialize = {index:value for index, value in enumerate(inv_data['Life (yr)'])})
        self.model.mu_e = Param(self.model.I, initialize = {index:float(co2_scaling['value'].iloc[index+1] * other_params['co2_lim (ktCO2)'].iloc[0]) for index, value in enumerate(self.model.I)})
        self.model.mu_dp = Param(self.model.I, initialize = {index:float(demand_scaling['value'].iloc[index+1] * other_params['demand_scaling'].iloc[0]) for index, value in enumerate(self.model.I)})
        self.model.c_co2 = Param(self.model.I, initialize = {index:other_params['co2_cost (mn£/ktCO2)'].iloc[0] for index, value in enumerate(self.model.I)})

        # operational self.model parameters
        
        weight = (17520/((len(self.model.T_n[0]))))
        
        self.model.W = Param(self.model.T, initialize = {index:weight for index, value in enumerate(self.model.T)})
        self.model.H = Param(self.model.T, initialize = {index:other_params['hour'].iloc[0] for index, value in enumerate(self.model.T)})
        self.model.alpha_G = Param(self.model.G, initialize = {index:value for index, value in enumerate(conventional_param['ramping (GW/GW)'])})
        
        if self.num_short_term_scenario == 1:
            wind_scenarios = scenarios[0,:].flatten()
            sun_scenarios = scenarios[1,:].flatten()
            demand_scenarios = scenarios[2,:].flatten()
        else:
            wind_scenarios = scenarios[:,0,:].flatten()
            sun_scenarios = scenarios[:,1,:].flatten()
            demand_scenarios = scenarios[:,2,:].flatten()
        
        r_dict = {}
        demand_dict = {}
        
        for t in self.model.T:
            
            for r in self.model.R:
                if r == 0 or r == 1:
                    r_dict[(r, t)] = wind_scenarios[t]
                else:
                    r_dict[(r, t)] = sun_scenarios[t]
            
            demand_dict[t] = demand_scenarios[t]

                
                        
           
        self.model.P_DP = Param(self.model.T, initialize = demand_dict)                
        self.model.R_r = Param( self.model.R, self.model.T, initialize = r_dict)
        self.model.eta_se = Param(self.model.S, initialize = {index:value for index, value in enumerate(storage_param['efficiency'])})
        self.model.gamma_se = Param(self.model.S, initialize = {index:value for index, value in enumerate(storage_param['power_ratio (GWh/GW)'])})
        self.model.E_g = Param(self.model.G, initialize = {index:value for index, value in enumerate(conventional_param['CO2_emissions (ktco2/GWh)'])})
        self.model.efficiency = Param(self.model.G, initialize = {index:value for index, value in enumerate(conventional_param['efficiency'])})
        
        total_Gcost = {}
        for i in self.model.I:
            for g, r in conventional_param.iterrows():
                total_Gcost[(i,g)] = (r['VarOM (mn£/GWh)']) + (r['FuelCost (mn£/GWh)'] / self.model.efficiency[g]) + ((r['CO2_emissions (ktco2/GWh)'] ) * (self.model.c_co2[i] / self.model.efficiency[g]))
        
        self.model.C_g = Param(self.model.I, self.model.G, initialize = total_Gcost)
        self.model.C_se = Param(self.model.S, initialize = {index:value for index, value in enumerate(storage_param['VarOM (mn£/GWh)'])})
        self.model.C_shed_p = Param(initialize =  other_params['demand_shedding (mn£/GW)'].iloc[0])
        

        # investment planning self.model variables
        self.model.x_acc = Var(self.model.P, self.model.I, domain=NonNegativeReals)
        self.model.x_inst = Var(self.model.P, self.model.I_0, domain=NonNegativeReals)

        # operational self.model variables
        self.model.p_accG = Var(self.model.G, self.model.I, domain=NonNegativeReals)
        self.model.p_accR = Var(self.model.R, self.model.I, domain=NonNegativeReals)
        self.model.p_accSE = Var(self.model.S, self.model.I, domain=NonNegativeReals)
        self.model.p_G = Var(self.model.G, self.model.T, self.model.I, domain=NonNegativeReals)
        self.model.p_SEp = Var(self.model.S, self.model.T, self.model.I, domain=NonNegativeReals)
        self.model.p_SEm = Var(self.model.S, self.model.T, self.model.I, domain=NonNegativeReals)
        self.model.p_GshedP = Var(self.model.T, self.model.I, domain=NonNegativeReals)
        self.model.q_SE = Var(self.model.S,  self.model.T, self.model.I, domain=NonNegativeReals)
        self.model.p_ShedP = Var(self.model.T, self.model.I, domain=NonNegativeReals)
        

        # investment constraints

        def accumulated_capacity_constraint_rule(m, p, i):

            return m.x_acc[p, i] == m.X_init[p, i] + sum(m.x_inst[p, j] for j in m.I_i[i])

        self.model.accumulated_capacity_constraint = Constraint(self.model.P, self.model.I, rule=accumulated_capacity_constraint_rule)

        def max_capacity_constraint_rule(m, p, i):
            return m.x_acc[p, i] <= m.X_max[p]

        self.model.max_capacity_constraint = Constraint(self.model.P, self.model.I, rule=max_capacity_constraint_rule)
        
        def max_build_capacity_constraint_rule(m,i):
            
            return sum(self.model.x_inst[p,i] for p in self.model.P) <= 77.5  #max built capacity 15.5gw * 5 years calculated from atkins realis report https://www.atkinsrealis.com/en/media/trade-releases/2024/2024-01-16
        
        self.model.max_build_capacity_constraint = Constraint(self.model.I_0, rule= max_build_capacity_constraint_rule)
        
        # operational constraints

        def generator_capacity_limit_constraint_rule(m, g, t, i):

            return m.p_G[g, t, i] <= m.p_accG[g, i]

        self.model.generator_capacity_limit_constraint = Constraint(self.model.G, self.model.T, self.model.I,
                                                            rule=generator_capacity_limit_constraint_rule)

        def charging_power_constraint_rule(m, s, t, i):
            return m.p_SEp[s, t, i] <= m.p_accSE[s, i]

        self.model.charging_power_constraint = Constraint(self.model.S, self.model.T, self.model.I, rule=charging_power_constraint_rule)

        def discharging_power_constraint_rule(m, s, t, i):
            return m.p_SEm[s, t, i] <= m.p_accSE[s,i]

        self.model.discharging_power_constraint = Constraint(self.model.S, self.model.T, self.model.I, rule=discharging_power_constraint_rule)

        def energy_storage_level_constraint_rule(m, s, t, i):
            return m.q_SE[s, t, i] <= m.gamma_se[s] * m.p_accSE[s,i]

        self.model.energy_storage_level_constraint = Constraint(self.model.S, self.model.T, self.model.I,
                                                        rule=energy_storage_level_constraint_rule)

        def ramp_up_constraint_rule(m, g, i, T):
        
            constraint_list = []
            for t in T:
                if t is not min(T):
                    constraint_list.append(m.p_G[g, t,i] - m.p_G[g, t - 1,i] <= m.alpha_G[g] * m.H[t] * m.p_accG[g,i])
                    constraint_list.append(m.p_G[g, t,i] - m.p_G[g, t - 1,i] >= -m.alpha_G[g] * m.H[t] * m.p_accG[g,i])
                else:
                    constraint_list.append(m.p_G[g, t,i] == m.p_G[g, max(T),i])
            return constraint_list 

        
        def add_ramp_up_constraints(model):
            model.RampUpConstraint = ConstraintList()
            for i in model.I:
                for g in model.G:
                    for n in model.N:
                        for constraint in ramp_up_constraint_rule(model, g, i, model.T_n[n]):
                            model.RampUpConstraint.add(constraint)
                            
        add_ramp_up_constraints(self.model)
        
        
        def balance_constraint_rule(m, t, i):

            gen_power = sum(m.p_G[g, t, i] for g in m.G)
            discharge = sum(m.p_SEm[s, t, i] for s in m.S)
            renewables = sum((m.R_r[ r, t] * m.p_accR[r,i]) for r in m.R)
            charge = sum(m.p_SEp[s, t, i] for s in m.S)

            return gen_power + discharge + renewables + m.p_ShedP[t,i] ==  -(m.mu_dp[i] * m.P_DP[t]) + charge + m.p_GshedP[t, i]

        self.model.balance_constraint = Constraint(self.model.T, self.model.I, rule=balance_constraint_rule)
        
        def state_of_charge_constraint_rule(m, s, i, T):
            
            constraint_list = []
            for t in T:
                if t is not max(T):
                    constraint_list.append(m.q_SE[s, t + 1, i] == m.q_SE[s, t, i] + m.H[t] * (m.eta_se[s] * m.p_SEp[s, t, i] - m.p_SEm[s, t, i]))
                else:
                    constraint_list.append(m.q_SE[s, t, i] == m.q_SE[s, min(T), i])
            return constraint_list

        def add_state_of_charge_constraints(model):
            model.state_of_charge_constraint = ConstraintList()
            for i in model.I:
                for s in model.S:
                    for n in model.N:
                        for constraint in state_of_charge_constraint_rule(model, s, i, model.T_n[n]):
                            model.state_of_charge_constraint.add(constraint)

        add_state_of_charge_constraints(self.model)
        
        def initial_energy_level_rule(m, s, i, T):
            
            return m.q_SE[s,min(T), i] == ((m.p_accSE[s,i] * m.gamma_se[s])/2)

        def add_initial_energy_level_constraints(model):
            model.initial_energy_level_constraint = ConstraintList()
            for i in model.I:
                for s in model.S:
                    for n in model.N:
                        
                        model.initial_energy_level_constraint.add(initial_energy_level_rule(model, s, i, model.T_n[n]))

        add_initial_energy_level_constraints(self.model)
        

        def co2_constraint_rule(m, i, T):
            constraint_list = []
            for t in T:
                constraint_list.append((sum((m.H[t] * (m.E_g[g]/ m.efficiency[g]) * (m.p_G[g, t, i] )) for g in m.G))/len(self.model.T) <= (m.mu_e[i]/17520))
            return constraint_list
        
        def add_co2_constraint(model):
            model.co2_constraint = ConstraintList()
            for i in model.I:
                for n in model.N:
                    
                    model.co2_constraint.add(
                        sum((sum(model.W[t] * model.H[t] * (model.E_g[g] / model.efficiency[g]) * model.p_G[g, t, i] 
                            for g in model.G)) for t in model.T_n[n])
                        <= model.mu_e[i] 
                    )

        add_co2_constraint(self.model)
        
        #self.model.co2_constraint = Constraint(self.model.I, rule=co2_constraint_rule)
        
        # constraints for operational problem coefficient passed from the investment problem
        
        def generator_rule(m, p, g, i):
            return m.x_acc[p, i]  == m.p_accG[p,i]
        
        def storage_rule(m,p,s,i):
            return m.x_acc[p,i]  == m.p_accSE[p-6,i]
        
        def renewable_rule(m, p, r, i):
            return m.x_acc[p,i]  == m.p_accR[p-8,i]
        
        self.model.technology_to_generator = Constraint(self.model.P_to_G, self.model.G, self.model.I, rule=generator_rule)
        self.model.technology_to_storage = Constraint(self.model.P_to_S, self.model.S, self.model.I, rule=storage_rule)
        self.model.technology_to_renewable = Constraint(self.model.P_to_R, self.model.R, self.model.I, rule=renewable_rule)
        
        # objective function

        def objective_rule(m):
            
            inv_cost = sum( ((m.delta_I_0[i]*m.pi_I_0[i]) * sum((m.C_inv[p,i]*m.x_inst[p,i]) for p in m.P)) for i in m.I_0)
            fix_cost = sum(((m.delta_I[i]*m.pi_I[i]) * sum((m.C_fix[p,i]*m.x_acc[p,i]) for p in m.P)) for i in m.I)
            
            C_inv = inv_cost + m.k*fix_cost                
            
            return (
                C_inv
                + m.k * sum( m.pi_I[i] *
                    sum(  m.pi_n[n] *
                        sum( 
                            (
                                m.W[t] * m.H[t]
                            ) * (
                                sum(m.C_g[i, g] * m.p_G[g, t, i] for g in m.G)
                                + sum(m.C_se[s] * m.p_SEp[s, t, i] for s in m.S)
                                + m.C_shed_p * m.p_ShedP[t, i]
                            )
                            for t in m.T_n[n]
                        )
                        for n in m.N
                    )
                    for i in m.I
                )
            )

        
        self.model.objective = Objective(rule= objective_rule, sense= minimize)

        return self.model

    
    def get_model(self):
        return self.model
    
    