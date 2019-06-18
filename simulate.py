# simulate.py
# Suhas Vijaykumar, June 2019

# This file contains the code that simulates the dgp given
# dgp parameters and an estimate of the optimal policy

from dgp import BasicModel, BasicModelAgent
from compute_policy import compute_policy, vf_iteration
import numpy as np
from numba import prange, jit, njit

import pickle

n_models = 100

def simulate_basic_model_slow(agents = 100, T = 1000):
	model = BasicModel()
	DynamicStates = np.empty(agents, T)
	StaticStates = np.empty(agents, p)
	Actions = np.empty(agents, T)

	for i in range(agents):
		agent = BasicModelAgent(model)
		_, policy, grid = compute_policy(agent, draws=10e4)
		agent_history = agent.simulate(policy, grid, T)
		DynamicStates[i,:] = agent_history[:,0]
		Actions[i,:] = agent_history[:,1]

	return {
		'static_states' : StaticStates,
		'dynamic_states' : DynamicStates,
		'actions' : Actions
		'environment_unobservables' : model
		'individual_unobservables' : agent
	}

if __name__=='__main__':
	models = [simulate_basic_model_slow() for _ in range(n_models)]
	pickle.dump(models,open('data_file.pkl','wb'))

# @njit
# def simulate_basic_model_fast(agents=100, B0, B1, A0, A1, beta, p):

# 	DynamicStates = np.empty(agents, T, 1)
# 	StaticStates = np.empty(agents, T, p)
# 	Actions = np.empty(agents, T, 1)

# 	AR0s = np.empty(agents)
# 	AR1s = np.empty(agents)
# 	C0s = np.empty(agents)
# 	C1s = np.empty(agents)
# 	D0s = np.empty(agents)
# 	D1s = np.empty(agents)

# 	grid=np.linspace(-200,200,10e4)
# 	policies = np.empty(agents,10e4)

# 	for i in prange(agents):

# 		static_state = BasicModelAgent.generate_static_state(1000)
# 		initial_dynamic_state = BasicModelAgent.generate_dynamic_state(1)

# 		StaticStates[i,0,:] = static_state
# 		DynamicStates[i,0,1] = initial_dynamic_state

# 		AR0s[i] = np.dot(A0,static_state)
# 		AR1s[i] = np.dot(A1,static_state)
# 		C0s[i] = np.dot(B0[1:],static_state)
# 		C1s[i] = np.dot(B1[1:],static_state)
# 		D0s[i] = B0[0]
# 		D1s[i] = B1[0]

# 		_, policy, _ = vf_iteration(
# 			X = grid,
# 			TX0 = BasicModelAgent.fast_basic_next_state(AR0s[i],AR1s[i],grid,0,0),
# 			TX1 = BasicModelAgent.fast_basic_next_state(AR0s[i],AR1s[i],grid,1,0),
# 			EU0 = BasicModelAgent.fast_basic_util(C0s[i],C1s[i],D0s[i],D1s[i],grid,0,0),
# 			EU1 = BasicModelAgent.fast_basic_util(C0s[i],C1s[i],D0s[i],D1s[i],grid,1,0),
# 			b = beta,
# 			n_draws = 10e4,
# 			precision = 1e-6,
# 			verbose = False
# 		)

# 		for t in range(T):
# 			actions






		
		






	
	
	