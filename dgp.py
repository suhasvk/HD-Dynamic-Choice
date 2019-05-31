# dgp.py
# Suhas Vijaykumar, May 2019

# This file will contain the code used to simulate the
# data generating processes used to evaluate proposed
# estimators

import numpy as np 
import random

# Set the random seed for replicability
RANDOM_SEED = 13

# Set the sample size
n = 1000

# Set the dimension of the (high-dimensional, static) component of the state
p = 1000

# This function will generate an approximately sparse signal of the form Theta_s + Theta_d
# - p = dimension of vector
# - s = size of support of Theta_s
# - Theta_d has k'th coordinate U_k/k^2 where U_k is Unif[-1,1].

def approximately_sparse(p, s=10, true_sparse_component=True, decay_component=True, seed=RANDOM_SEED):
 	
	dense_part = np.multiply(
		np.random.random_sample(size=(p,)),
		np.array(map(lambda x: float(1)/(x**2), range(1,p+1)))
	)

	sparse_support = np.random.choice(p,s)
	sparse_coefficients = np.random.random_sample(size=(s,))

	sparse_part = np.zeros(p)
 	sparse_part[sparse_support] = sparse_coefficients

 	return dense_part + sparse_part