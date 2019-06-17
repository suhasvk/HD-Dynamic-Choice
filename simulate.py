# simulate.py
# Suhas Vijaykumar, June 2019

# This file contains the code that simulates the dgp given
# dgp parameters and an estimate of the optimal policy

from dgp import BasicModel, BasicModelAgent
from compute_policy import compute_policy
import numpy as np
from numba import prange, jit, njit

@jit(nopython=False)
def simulate_basic_model(agents=100, T=100):
	model = BasicModel()
	DynamicStates = np.empty(agents, T, model.q)
	StaticStates = np.empty(agents, T, model.p)
	Actions = np.empty(agents, T, 1)
	for i in prange(agents):
		agent = BasicModelAgent(model)
		policy, value_fn = compute_policy(agent, method='vf_iteration')
		


	
	
	