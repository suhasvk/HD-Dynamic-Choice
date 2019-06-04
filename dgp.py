# dgp.py
# Suhas Vijaykumar, May 2019

# This file will contain the code used to simulate the
# data generating processes used to evaluate proposed
# estimators

import numpy as np 
import matplotlib.pyplot as mpl

from scipy.interpolate import interp1d
from scipy.stats import norm as normal

from vf_iteration import compute_value_function
from simulate import simulate_dgp
# Set the random seed for replicability
RANDOM_SEED = 13
np.random.seed(RANDOM_SEED)

# Set the sample size
n = 1000

# Set the dimension of the (high-dimensional, static) component of the state
p = 1000

# Set the size of support of exactly sparse vectors
s = 10

# Set the discount parameter
beta = 0.1

# This function will generate an approximately sparse signal of the form Theta_1 + Theta_2
# - p = dimension of vector
# - s = size of support of (truly sparse) Theta_1
# - Theta_2 (apx. sparse) has k'th coordinate U_k/k^2 where U_k is Unif[-1,1].
def approximately_sparse(p, s=s, true_sparse_component=True, decay_component=True):
 	
	dense_part = np.multiply(
		1 - 2 * np.random.random_sample(size=(p,)),
		1./np.power(np.array(range(1,p+1)),2)
	)

	sparse_support = np.random.choice(p,s)
	sparse_coefficients = 1 - 2 * np.random.random_sample(size=(s,))

	sparse_part = np.zeros(p)
	sparse_part[sparse_support] = sparse_coefficients

	return dense_part*int(decay_component) + sparse_part*int(true_sparse_component)

# This generates the (observable) state of the agent
def generate_observable_parameters(n=n,p=p):

	# This is the dynamic state (drawn from a standard normal distribution)
	ThetaD = np.random.randn(1)

	# This is the static state (high dimensional, binary/categorical)
	ThetaS = np.random.randint(2,size=(p,))

	return np.hstack((ThetaD,ThetaS))

# This generates the (unobserbable) utility function and state evolution parameters
def generate_unobservable_parameters(n=n,p=p):

	# These determine the state-specific utility function
	B0 = approximately_sparse(p+1)
	B1 = approximately_sparse(p+1)

	# These are the multipliers that determine  AR coefficients for the dynamic state
	# We divide by s+1 to ensure that AR coefficients lie strictly between zero and one
	A0 = np.absolute(approximately_sparse(p, decay_component=False)) / (s+1) + 1./np.power(p,2)
	A1 = np.absolute(approximately_sparse(p, decay_component=False)) / (s+1) + 1./np.power(p,2)

	return (A0,A1,B0,B1)

def test():
	Theta = generate_observable_parameters()

	Unobservables = generate_unobservable_parameters()

	Policy = compute_value_function(Theta, Unobservables)

	Data = simulate_dgp(Policy, Theta, Unobservables)

if __name__ == '__main__':
	test()


	