# dgp.py
# Suhas Vijaykumar, May 2019

# This file will contain the code used to simulate the
# data generating processes used to evaluate proposed
# estimators

import numpy as np 
import random

# Set the random seed for replicability
RANDOM_SEED = 13
numpy.random.seed(RANDOM_SEED)

# Set the sample size
n = 1000

# Set the dimension of the (high-dimensional, static) component of the state
p = 1000

def compute_value_function(grid, Theta, Unobservables):

	pass

# This function will generate an approximately sparse signal of the form Theta_s + Theta_d
# - p = dimension of vector
# - s = size of support of Theta_s
# - Theta_d has k'th coordinate U_k/k^2 where U_k is Unif[-1,1].
def approximately_sparse(p, s=10, true_sparse_component=True, decay_component=True):
 	
	dense_part = np.multiply(
		1 - 2 * np.random.random_sample(size=(p,)),
		np.array(map(lambda x: float(1)/(x**2), range(1,p+1)))
	)

	sparse_support = np.random.choice(p,s)
	sparse_coefficients = 1 - 2 * np.random.random_sample(size=(s,))

	sparse_part = np.zeros(p)
 	sparse_part[sparse_support] = sparse_coefficients

 	return dense_part + sparse_par

# This generates the (observable) state of the agent
def generate_observable_parameters(n,p):

	# This is the dynamic state (drawn from a standard normal distribution)
	ThetaD = np.random.randn(1)

	# This is the static state (high dimensional, binary/categorical)
	ThetaS = np.random.randint(2,size=(p,))

	return np.hstack((Theta0,ThetaS))

# This generates the (unobserbable) utility function and state evolution parameters
def generate_unobservable_parameters(n,p,sigma_epsilon_utility, sigma_epsilon_state):

	# These determine the state-specific utility function
	B0 = approximately_sparse(p+1)
	B1 = approximately_sparse(p+1)

	# These are the coefficients that determine the AR coefficients for the dynamic state
	A0 = approximately_sparse(p)
	A1 = approximately_sparse(p)

	return (A0,A1,B0,B1)


	