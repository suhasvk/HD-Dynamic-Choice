# compute_policy.py
# Suhas Vijaykumar, June 2019
# This file contains the value function iteration procedure.
# Requires debugging.

import numpy as np
from scipy.interpolate import interpn, interp1d
from numba import prange

def compute_policy_dummy(model_instance, grid=np.linspace(-200,200,10e4)):
	return {"value_function": np.array([0 for _ in grid]), "policy": grid > 0, "grid": grid}
	
def compute_policy_q_learning(model_instance):
	pass
	
def compute_policy_vf_iteration(model_agent, X=np.linspace(-200,200,10e4), precision=1e-6, ε_vector=np.random.randn(10e5)):
	
	expected_utility = np.vectorize(lambda a, x : model_agent.utility(dynamic_state=x, action=x, shock=0))
	transition_kernel = lambda a, x, ε : model_agent.next_state(dynamic_state=x, action=a, shock=ε)
	
	
	β = model_agent.discount_factor
	
	
	δ = lambda v1, v2: np.amax(np.abs(v1-v2))
	
	def bellman_operator(Vf):
		TV = np.empty_like(grid)
		for i in prange(grid.shape[0]):
			x = grid[i,:]
			t1 = lambda ε : transition(1, x, ε)
			t0 = lambda ε : transition(0, x, ε)
			TV[i,:] = max(
				EU(1,x) + β * np.mean(V(t1(ε_vector))),
				EU(0,x) + β * np.mean(V(t0(ε_vector)))
			) 
		return TV	

	# Compute the initial value function values
	V_old = np.empty_like(grid)
	V_new = bellman_operator(lambda x: 0)

	while δ(V_new,V_old) > precision:
		V_old = V_new
		Vf = interp1d(grid,V_old,assume_sorted=True)
		V_new = bellman_operator

	# VF contains our final estimate of the value function
	VF = V_new

	# Policy contains the optimal action for each value in the grid
	Policy = C0 + np.multiply(D0,grid) + beta*CV0 < np.multiply(D1,grid) + beta*CV1
	
	return {"value_function": VF, "policy": policy, "grid": grid}
