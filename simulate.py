# simulate.py
# Suhas Vijaykumar, June 2019

# This file contains the code that simulates the dgp given
# dgp parameters and an estimate of the optimal policy

import numpy as np

class ModelAgent(object):
	
	def __init__(self, model):
		self.model = model 
		self.static_state = generate_static_state()
		self.dynamic_state = generate_dynamic_state()
		self.initial_dynamic_state = self.dynamic_state
		
	def generate_static_state(self):
		return np.random.randn(self.model.p)
		
	def generate_dynamic_state(self)
		return np.random.randn(self.model.q)
		
	def utility(self, dynamic_state=self.dynamic_state, action, shock):
		state = np.concatenate((dynamic_state,self.static_state))
		return self.model.utility(state, action, shock)
		
	def next_state(self, dynamic_state=self.dynamic_state, action, shock, update=True):
		state = np.concatenate((dynamic_state,self.static_state))
		next_state = self.model.next_state(state, action, shock)
		self.dynamic_state = (next_state if update else self.dynamic_state)
		return next_state
		
	def reset_state(self):
		self.dynamic_state = self.initial_dynamic_state
		
	def simulate(self, n=self.model.n, shocks, policy):
		states = np.zeros(n,self.model.q)
		actions = np.zeros(n)
		for i in range(n):
			states[i,:] = self.dynamic_state
			actions[i] = policy(self.dynamic_state)
			self.next_state(actions[i],shocks[i])
		self.reset_state()
		return (states, actions)
			
def BasicModelAgent(ModelInstance):
	
	def __init__(self, *args, **kwargs):
		super(BasicModelInstance, self).__init__(*args, **kwargs)
		
		self.AR1 = np.dot(self.model.A1, self.static_state)
		self.AR0 = np.dot(self.model.A0, self.static_state)
		
		self.C1 = np.dot(self.model.B1[1:], self.static_state)
		self.C0 = np.dot(self.model.B0[1:], self.static_state)
		
		self.D1 = self.model.B1[0]
		self.D0 = self.model.B0[0]
		
	def next_state(self, *args, **kwargs):
		next_state = (action*self.AR1 + (1-action)*self.AR0) * dynamic_state + shock
		self.dynamic_state = (next_state if update else self.dynamic_state)
		return next_state

	def utility(self, *args, **kwargs):
		return (action * (self.C1 + self.D1 * dynamic_state) + (1-action) * (self.C0 + self.D0 * dynamic_state))		


	
	
	