from pyomo.environ import *
import pandas as pd
import numpy as np

class OperationalSubProblem:
    
    def __init__(self, scenarios, num_short_term_scenario):
        self.model = ConcreteModel()
        self.scenarios = scenarios
        self.num_short_term_scenario = num_short_term_scenario
        self.build_model()
        

    def build_model(self):
        
        inv_data = pd.read_csv('cleaned/single_investment_data.csv')
        conventional_param = pd.read_csv('cleaned/conventional_params.csv')
        storage_param = pd.read_csv('cleaned/storage_params.csv')
        other_params = pd.read_csv('cleaned/other_params.csv')
        co2_budget_scaling = pd.read_csv('cleaned/co2_budget_scaling_factor.csv')
        demand_scaling = pd.read_csv('cleaned/demand_scaling_factor.csv')
        
        scenarios_infos = pd.read_csv('scenarios/scenarios_info.csv')
        
        
        scenarios = self.scenarios

         

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

        # investment parameters passed to the operational probllem in addition to the X_acc variables
        self.model.mu_e = Param( initialize = 51075.904, mutable = True)
        self.model.mu_dp = Param( initialize = 1, mutable = True)
        self.model.c_co2 = Param( initialize = 0.018, mutable = True)

        # operational self.model parameters
        
        weight = (17520/((len(self.model.T))))
        
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
        
        self.total_Gcost = {}
        for g, r in conventional_param.iterrows():
            self.total_Gcost[g] = (r['VarOM (mn£/GWh)']) + (r['FuelCost (mn£/GWh)'] / self.model.efficiency[g]) + ((r['CO2_emissions (ktco2/GWh)'] ) * (self.model.c_co2 / self.model.efficiency[g]))

        self.model.C_g = Param(self.model.G, initialize = self.total_Gcost)
        self.model.C_se = Param(self.model.S, initialize = {index:value for index, value in enumerate(storage_param['VarOM (mn£/GWh)'])})
        self.model.C_shed_p = Param(initialize = other_params['demand_shedding (mn£/GW)'].iloc[0])
        

        # operational self.model variables
        self.model.p_accG = Var(self.model.G,  domain=NonNegativeReals)
        self.model.p_accR = Var(self.model.R,  domain=NonNegativeReals)
        self.model.p_accSE = Var(self.model.S,  domain=NonNegativeReals)
        self.model.p_G = Var(self.model.G, self.model.T,  domain=NonNegativeReals)
        self.model.p_SEp = Var(self.model.S, self.model.T,  domain=NonNegativeReals)
        self.model.p_SEm = Var(self.model.S, self.model.T,  domain=NonNegativeReals)
        self.model.p_GshedP = Var(self.model.T,  domain=NonNegativeReals)
        self.model.q_SE = Var(self.model.S,  self.model.T,  domain=NonNegativeReals)
        self.model.p_ShedP = Var(self.model.T,  domain=NonNegativeReals)

        # operational constraints

        def ramp_up_constraint_rule(m, g, T):
        
            constraint_list = []
            for t in T:
                if t is not min(T):
                    constraint_list.append(m.p_G[g, t] - m.p_G[g, t - 1] <= m.alpha_G[g] * m.H[t] * m.p_accG[g])
                    constraint_list.append(m.p_G[g, t] - m.p_G[g, t - 1] >= -m.alpha_G[g] * m.H[t] * m.p_accG[g])
                else:
                    constraint_list.append(m.p_G[g, t] == m.p_G[g, max(T)])
            return constraint_list 

        
        def add_ramp_up_constraints(model):
            model.RampUpConstraint = ConstraintList()
           
            for g in model.G:
                for n in model.N:
                    for constraint in ramp_up_constraint_rule(model, g, model.T_n[n]):
                        model.RampUpConstraint.add(constraint)
                            
        add_ramp_up_constraints(self.model)
        
        
        def balance_constraint_rule(m, t):

            gen_power = sum(m.p_G[g, t] for g in m.G)
            discharge = sum(m.p_SEm[s, t] for s in m.S)
            renewables = sum((m.R_r[ r, t] * m.p_accR[r]) for r in m.R)
            charge = sum(m.p_SEp[s, t] for s in m.S)

            return gen_power + discharge + renewables + m.p_ShedP[t] ==  -(m.mu_dp * m.P_DP[t]) + charge + m.p_GshedP[t]

        self.model.balance_constraint = Constraint(self.model.T, rule=balance_constraint_rule)
        
        def state_of_charge_constraint_rule(m, s, T):
            
            constraint_list = []
            for t in T:
                if t is not max(T):
                    constraint_list.append(m.q_SE[s, t + 1] == m.q_SE[s, t] + m.H[t] * (m.eta_se[s] * m.p_SEp[s, t] - m.p_SEm[s, t]))
                else:
                    constraint_list.append(m.q_SE[s, t] == m.q_SE[s, min(T)])
            return constraint_list

        def add_state_of_charge_constraints(model):
            model.state_of_charge_constraint = ConstraintList()
            
            for s in model.S:
                for n in model.N:
                    for constraint in state_of_charge_constraint_rule(model, s, model.T_n[n]):
                        model.state_of_charge_constraint.add(constraint)

        add_state_of_charge_constraints(self.model)
        
        def initial_energy_level_rule(m, s, T):
            
            return m.q_SE[s,min(T)] == ((m.p_accSE[s] * m.gamma_se[s])/2)

        def add_initial_energy_level_constraints(model):
            model.initial_energy_level_constraint = ConstraintList()
            
            for s in model.S:
                for n in model.N:
                    
                    model.initial_energy_level_constraint.add(initial_energy_level_rule(model, s, model.T_n[n]))

        add_initial_energy_level_constraints(self.model)
        
        

        def add_co2_constraint(model):
            model.co2_constraint = ConstraintList()
            
            for n in model.N:
                
                model.co2_constraint.add(
                    sum((sum(model.W[t] * model.H[t] * (model.E_g[g] / model.efficiency[g]) * model.p_G[g, t] 
                        for g in model.G)) for t in model.T_n[n])
                    <= model.mu_e 
                )

        add_co2_constraint(self.model)
        
        
        # objective function

        def objective_rule(m):
            return  (sum(
                            (
                                m.W[t] * m.H[t]
                            ) * (
                                sum(m.C_g[ g] * m.p_G[g, t] for g in m.G)
                                + sum(m.C_se[s] * m.p_SEp[s, t] for s in m.S)
                                + m.C_shed_p * m.p_ShedP[t]
                            )
                            for t in m.T
                        ))
                                       
        
        self.model.objective = Objective(rule= objective_rule, sense= minimize)

        return self.model

    
    def get_model(self):
        return self.model
    
    