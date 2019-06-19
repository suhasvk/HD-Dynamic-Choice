# compute_policy.py
# encoding: utf-8
# Suhas Vijaykumar, June 2019
# This file contains the value function iteration procedure.
# Requires debugging.

import numpy as np
from numba import prange, njit

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as mpl

def compute_policy_dummy(model_instance, grid=np.linspace(-200,200,10e4)):
	return np.zeros_like(grid), grid > 0, grid
	
def q_learning(model_instance):
	pass	

# This computes an approximation to the Bellman operator obtained by substituting the 
# normal distribution with the empirical distribution of 10e3 draws from the normal 
# distribution (by Monte Carlo integration) 
@njit
def bellman_operator(V, X, TX0, TX1, EU0, EU1, random_draws, b):
	TV = np.empty_like(V)
	Policy = np.empty_like(V)
	for i in prange(X.shape[0]):
		x = X[i]

		V1 = np.mean( np.interp(TX1[i] + random_draws, X, V) )
		V0 = np.mean( np.interp(TX0[i] + random_draws, X, V) )
	
		TV[i] = np.maximum(
			EU1[i] + b * V1,
			EU0[i] + b * V0
		) 
		Policy[i] = np.greater_equal(
			EU1[i] + b * V1,  
			EU0[i] + b * V0
		)
	return TV, Policy

@njit 
def vf_iteration(X, TX0, TX1, EU0, EU1, b, n_draws, precision, verbose):

	if verbose:
		print("computing value function by vf_iteration")
	V0 = np.zeros(X.shape[0])
	V1 = np.fmax(EU0, EU1)
	random_draws = np.random.randn(n_draws)

	# The uniform distance between two piecewise-linear approximate value functions 
	distance = lambda v1, v2: np.amax(np.abs(v1-v2))

	iterations = 0
	while distance(V1,V0) > precision:
		V0 = V1
		V1, Policy = bellman_operator(V0, X, TX0, TX1, EU0, EU1, random_draws, b)
		iterations += 1
		if verbose:
			print("iterations:", iterations)
			print("distance:", distance(V1,V0))

	return V1, Policy, X

def compute_policy(model_agent, grid=np.linspace(-200,200,10e4), precision=1e-6, n_draws = 1e3, verbose=False, method = 'vf_iteration'):
	
	expected_utility = np.vectorize(lambda a, x : model_agent.utility(action=a, shock=0, dynamic_state=x))
	transition_kernel = np.vectorize(lambda a, x : model_agent.next_state(action=a, shock=0, dynamic_state=x))

	# The agent's discount factor is given by b	
	b = model_agent.model.discount_factor

	# TX0 = transition_kernel(grid,0)
	# print(TX0.shape)
	# print(TX0[10])

	if method is 'vf_iteration':
		ValueFunction, OptimalPolicy, Grid = vf_iteration(
			X = grid,
			TX0 = transition_kernel(grid,0),
			TX1 = transition_kernel(grid,1),
			EU0 = expected_utility(grid,0),
			EU1 = expected_utility(grid,1),
			b = b,
			n_draws = n_draws,
			precision = precision,
			verbose = verbose
		) 

	else:
		raise Exception("Dynamic programming method %s not supported!" % method)

	return (ValueFunction, OptimalPolicy, Grid)

	
def test_vf_iteration():
	model = BasicModel()
	agent = BasicModelAgent(model)
	policy, V_final, grid = compute_policy(agent, verbose=True, method='vf_iteration')
	
if __name__ == '__main__':
	from dgp import *
	test_vf_iteration()	