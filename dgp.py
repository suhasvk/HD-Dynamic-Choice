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
		
		self.generate_model_parameters()
		
	def generate_model_parameters(self):
		pass
		
	def utility(self, action, state, shock):
		pass
		
	def next_dynamic_state(self, action, state, shock):
		pass
		
# 	def agent(self, static_state=None, dynamic_state=None):
# 		pass
			
# An instance of the BasicModel class is a data generating process with approximately sparse utility specification
# and truly sparse AR(1) law of motion.

class BasicModel(Model):
	
	def __init__(self, s=10, *args, **kwargs):
		self.s = s
		super(BasicModel, self).__init__(*args, **kwargs)
	
	def generate_model_parameters(self):
		
		# These determine the state- and action-specific utility function
		self.B0 = approximately_sparse(self.p+self.q, self.s)
		self.B1 = approximately_sparse(self.p+self.q, self.s)

		# These are the multipliers that determine AR coefficients for the dynamic state
		# We divide by s+1 to ensure that AR coefficients lie strictly between zero and one
		self.A0 = np.absolute(approximately_sparse(self.p, self.s, decay_component=False)) / (self.s+1) + 1./np.power(self.p,2)
		self.A1 = np.absolute(approximately_sparse(self.p, self.s, decay_component=False)) / (self.s+1) + 1./np.power(self.p,2)	
			
	def utility(self, action, state, shock):
		return action * np.dot(self.B1, state) + (1-action) * np.dot(self.B0, state) + shock
		
	def next_dynamic_state(self, action, state, shock):
 		return ( action * np.dot(self.A1,state[self.q:]) + (1-action)*np.dot(self.A0,state[self.q:]) ) * state[0:self.q] + shock
 		
#  	def agent(self, static_state=None, dynamic_state=None):
#  		new_agent = BasicModelAgent(self)
#  		print(new_agent)
#  		new_agent.static_state = (new_agent.static_state if static_state is None else static_state)
#  		new_agent.dynamic_state = (new_agent.dynamic_state if dynamic_state is None else dynamic_state)
#  		return new_agent
		 
# This helper function generates an approximately sparse signal of the form Theta_1 + Theta_2
# - p = dimension of vector
# - s = size of support of (truly sparse) Theta_1
# - Theta_2 (apx. sparse) has k'th coordinate U_k/k^2 where U_k is Unif[-1,1].
def approximately_sparse(p, s, true_sparse_component=True, decay_component=True):
 	
	dense_part = np.multiply(
		1 - 2 * np.random.random_sample(size=(p,)),
		1./np.power(np.array(range(1,p+1)),2)
	)

	sparse_support = np.random.choice(p,s)
	sparse_coefficients = 1 - 2 * np.random.random_sample(size=(s,))

	sparse_part = np.zeros(p)
	sparse_part[sparse_support] = sparse_coefficients

	return dense_part*int(decay_component) + sparse_part*int(true_sparse_component)

class ModelAgent(object):
	
	def __init__(self, model):

		self.model = model 
		self.static_state = self.generate_static_state()
		self.dynamic_state = self.generate_dynamic_state()
		self.initial_dynamic_state = self.dynamic_state
		
	def generate_static_state(self):
		return np.random.randn(self.model.p)
		
	def generate_dynamic_state(self):
		return np.random.randn(self.model.q)
		
	def utility(self, action, shock, dynamic_state=None):
		dynamic_state = (self.dynamic_state if dynamic_state is None else dynamic_state)
		state = np.concatenate((dynamic_state,self.static_state))
		return self.model.utility(state, action, shock)
		
	def next_state(self, action, shock, dynamic_state=None, update=True):
		dynamic_state = (self.dynamic_state if dynamic_state is None else dynamic_state)
		state = np.concatenate((dynamic_state,self.static_state))
		next_state = self.model.next_state(state, action, shock)
		self.dynamic_state = (next_state if update else self.dynamic_state)
		return next_state
		
	def reset_state(self):
		self.dynamic_state = self.initial_dynamic_state
		
	def simulate(self, shocks, policy, n=None):
		n = (self.model.n if n is None else n)
		states = np.zeros(n,self.model.q)
		actions = np.zeros(n)
		for i in range(n):
			states[i,:] = self.dynamic_state
			actions[i] = policy(self.dynamic_state)
			self.next_state(actions[i],shocks[i])
		self.reset_state()
		return (states, actions)
			
class BasicModelAgent(ModelAgent):
	
	def __init__(self, *args, **kwargs):

		super(BasicModelAgent, self).__init__( *args, **kwargs)	
		
		self.AR1 = np.dot(self.model.A1, self.static_state)
		self.AR0 = np.dot(self.model.A0, self.static_state)
		
		self.C1 = np.dot(self.model.B1[1:], self.static_state)
		self.C0 = np.dot(self.model.B0[1:], self.static_state)
		
		self.D1 = self.model.B1[0]
		self.D0 = self.model.B0[0]
		
	def next_state(self, action, shock, dynamic_state=None, update=True):
		next_state = (action*self.AR1 + (1-action)*self.AR0) * dynamic_state + shock
		self.dynamic_state = (next_state if update else self.dynamic_state)
		return next_state

	def utility(self, action, shock, dynamic_state=None):
		return (action * (self.C1 + self.D1 * dynamic_state) + (1-action) * (self.C0 + self.D0 * dynamic_state))		
		
# TESTS

def test_basic_model():
	m = BasicModel()
	print("type:", type(m))
	print(m.A0.shape, m.A1.shape, m.B0.shape, m.B1.shape)
	print(m.utility(0,np.zeros(1001),0))
	print(m.next_dynamic_state(0,np.zeros(1001),0))
	print(m.next_dynamic_state(0,np.zeros(1001),1))
	print(m.next_dynamic_state(1,np.zeros(1001),0))
	print(m.next_dynamic_state(1,np.array([1]+[0]*1000),0))
	print(m.next_dynamic_state(0,np.array([1]+[0]*1000),0))

def test_model_agent():
	m = BasicModel()
# 	a1 = ModelAgent(m)
# 	print "type1:", type(a1)
	a2 = BasicModelAgent(m)
	print("type2:", type(a2))
	
if __name__ == '__main__':
	test_basic_model()
	test_model_agent()