# dgp.py
# Suhas Vijaykumar, May 2019

# This file will contain the definitions of various models for the data generating process
# These definitions will contain all information necessary to solve and simulate the model

import numpy as np 

# The "Model" class is a template outlining the information required to solve and simulate a model within our framework
# Each instance of the Model class will be able to map states and actions to utilities and transition probabilities
# according to a specified data generating process.

class Model(object):
	
	def __init__(self, p=1000, q=1, beta=.01):
		
		self.q = q
		self.p = p
		self.discount_factor = beta
		
		self.generate_unobservable_parameters()
		
	def generate_model_parameters(self):
		pass
		
	def utility(self, action, state):
		pass
		
	def next_state(self, action, state, shock):
		pass
			
# An instance of the BasicModel class is a data generating process with approximately sparse utility specification
# and truly sparse AR(1) law of motion.

class BasicModel(Model):
	
	def __init__(self, s=1, *args, **kwargs):
		super(BasicModel, self).__init__(*args, **kwargs)
		self.s = s
	
	def generate_model_parameters(self):
		
		# These determine the state- and action-specific utility function
		self.B0 = approximately_sparse(self.p+1)
		self.B1 = approximately_sparse(self.p+1)

		# These are the multipliers that determine AR coefficients for the dynamic state
		# We divide by s+1 to ensure that AR coefficients lie strictly between zero and one
		self.A0 = np.absolute(approximately_sparse(self.p, decay_component=False)) / (self.s+1) + 1./np.power(self.p,2)
		self.A1 = np.absolute(approximately_sparse(self.p, decay_component=False)) / (self.s+1) + 1./np.power(self.p,2)	
			
	def utility(self, action, state, utility_shock):
		
		return action * np.dot(self.B1, state) + (1-action) * np.dot(self.B0, state) + utility_shock
		
	def next_dynamic_state(self, action, state, state_shock):
		
		return ( action * np.dot(self.A1,state[1:]) + (1-action)*np.dot(self.A0,state[1:]) ) * state[0] + state_shock
		
# This helper function generates an approximately sparse signal of the form Theta_1 + Theta_2
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

