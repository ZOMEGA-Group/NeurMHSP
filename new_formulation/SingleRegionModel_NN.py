import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from gurobi_ml import add_predictor_constr
from gurobi_ml.sklearn.preprocessing import add_standard_scaler_constr

class SingleRegionProblem_NN:
    
    def __init__(self, nn_model, scaler):
        self.model = gp.Model("SingleRegionModel_NN")
        self.nn_model = nn_model
        self.scaler = scaler
        self.build_model()

    def build_model(self):
        inv_data = pd.read_csv('cleaned/single_investment_data.csv')
        other_params = pd.read_csv('cleaned/other_params.csv')
        demand_scaling = pd.read_csv('cleaned/demand_scaling_factor.csv')
        co2_scaling = pd.read_csv('cleaned/co2_budget_scaling_factor.csv') 
        
        P = range(11)     
        I_0 = range(13)   
        I = range(12)     
        NN_inputs = range(14)
        
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

        
        self.C_inv = {(p, i): inv_data['CapeX (mn£/GW)'][p] for p in P for i in I_0}
        self.C_fix = {(p, i): inv_data['FixOM (mn£/GWyr)'][p] for p in P for i in I}
        X_max = {p: inv_data['Max_capacity (GW)'][p] for p in P}
        k = 5
        
       
        X_init = {}
        for p in P:
            for i in I:
                if i < 3:
                    X_init[(p, i)] = inv_data['capacity_at_5'][p]
                else:
                    X_init[(p, i)] = inv_data['capacity_at_10'][p]

        
        def calculate_discount_factor(period):
            return 1 / ((1 + 0.05) ** period)
        
        delta_i = {}
        for i in I:
            if i < 3:
                delta_i[i] = calculate_discount_factor(5)
            else:
                delta_i[i] = calculate_discount_factor(10)

        delta_i_0 = {}
        for i in I_0:
            if i == 0:
                delta_i_0[i] = calculate_discount_factor(0)
            elif i < 4:
                delta_i_0[i] = calculate_discount_factor(5)
            else:
                delta_i_0[i] = calculate_discount_factor(10)

        self.delta_I_0 = delta_i_0
        self.delta_I = delta_i

        self.pi_I_0 = {
            0: 1,
            1: 0.33,
            2: 0.33,
            3: 0.33,
            4: 0.11,
            5: 0.11,
            6: 0.11,
            7: 0.11,
            8: 0.11,
            9: 0.11,
            10: 0.11,
            11: 0.11,
            12: 0.11
        }

        self.pi_I = {i: (1/3 if i < 3 else 1/9) for i in I}

        mu_e = {i: float(co2_scaling['value'].iloc[i+1] *other_params['co2_lim (ktCO2)'].iloc[0]) for i in I}
        mu_dp = {i: float(demand_scaling['value'].iloc[i+1] *other_params['demand_scaling'].iloc[0]) for i in I}
        c_co2 = {i: float(other_params['co2_cost (mn£/ktCO2)'].iloc[0]) for i in I}

        
        
        x_acc = self.model.addVars(P, I, name="x_acc", lb=0)
        x_inst = self.model.addVars(P, I_0, name="x_inst", lb=0)
        ope_cost_surrogate = self.model.addVars(I, lb=0, ub=GRB.INFINITY, name="ope_cost_surrogate")
        nn_input = self.model.addVars(I, NN_inputs, lb=-GRB.INFINITY, name = 'nn_input')
        
        for p in P:
            for i in I:
                self.model.addConstr(
                    x_acc[p, i] == X_init[(p, i)] + gp.quicksum(x_inst[p, j] for j in ti_set[i]),
                    name=f"acc_capacity_p{p}_i{i}"
                )

        
        for p in P:
            for i in I:
                self.model.addConstr(
                    x_acc[p, i] <= X_max[p],
                    name=f"max_capacity_p{p}_i{i}"
                )
        
        for i in I_0:
            self.model.addConstr(gp.quicksum(x_inst[p, i] for p in P) <= 77.5, name=f"max_built_capacity_i{i}")

        
        inv_cost = gp.quicksum(
            self.delta_I_0[i] * self.pi_I_0[i] * gp.quicksum(self.C_inv[p, i] * x_inst[p, i] for p in P)
            for i in I_0
        )

        fix_cost = gp.quicksum(
            self.delta_I[i] * self.pi_I[i] * gp.quicksum(self.C_fix[p, i] * x_acc[p, i] for p in P)
            for i in I
        )
        
        operational_cost_term = gp.quicksum(self.pi_I[i] * ope_cost_surrogate[i] for i in I)
        
        self.model.setObjective(inv_cost + k * fix_cost + k * operational_cost_term, GRB.MINIMIZE)
        
        
        for i in I:
            for n in NN_inputs:
                if n in P:
                    self.model.addConstr(x_acc[n,i] == nn_input[i, n])
                
                if n == 11: 
                    self.model.addConstr(mu_e[i] == nn_input[i, n])
                if n == 12:
                    self.model.addConstr(c_co2[i] == nn_input[i, n]) 
                if n == 13:
                    self.model.addConstr(mu_dp[i] == nn_input[i, n])
                    
                 
        for i in I:
            input_vars_i = [nn_input[i,n] for n in NN_inputs]
            standard_scaler_constr = add_standard_scaler_constr(self.model, self.scaler, input_vars_i)
            pred_constr = add_predictor_constr(self.model, self.nn_model, standard_scaler_constr.output, ope_cost_surrogate[i])
        
        self.model.update()


    def get_model(self):
        return self.model
