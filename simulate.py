# simulate.py
# Suhas Vijaykumar, June 2019

# This file contains the code that simulates the dgp given
# dgp parameters and an estimate of the optimal policy

from scipy.interpolate import interp1d
import numpy as np

def simulate_dgp(Policy, Unobservables, theta_D, iterations=5000):
	
	# Get value function parameters
	A0, A1, B0, B1 = Unobservables
	
	# Reconstruct policy function from grid values by linear interpolation
	policy_values = Policy["policy"]
	policy_grid = Policy["grid"]
	policy_fn = interp1d(policy_values, policy_grid)
	
	actions = []
	states = []
	
	theta_D_current = theta_D
	
	# Now we simulate the actions and the resulting movement of the dynamic state according to
	# the policy function we computed
	
	for i in range(iterations):
	
		# Compute and record state and optimal action
		# Note that we round actions at the boundary; may be better to randomize
		states.append(theta_D_current)
		action = int(policy_fn(theta_D_current)) 
		actions.append(action)
		
		# Update state according to law of motion
		theta_D_current = (action * A1 + (1-action) * A0) * theta_D_current + np.random.randn(1)
		
	return {
		"actions": np.array(actions),
		"states":  np.array(states)
	}	
		
		
	
	
	
	
	