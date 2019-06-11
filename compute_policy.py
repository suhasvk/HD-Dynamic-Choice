# compute_policy.py
# Suhas Vijaykumar, June 2019
# This file contains the value function iteration procedure.
# Requires debugging.

def compute_policy_dummy(Theta, Unobservables, grid=np.linspace(-200,200,10e4), precision=1e-6, draws=10e5):
	return {"value_function": np.array([0 for _ in grid]), "policy": grid > 0, "grid": grid}

def compute_policy_vf_iteration(Theta, Unobservables, grid=np.linspace(-200,200,10e4), precision=1e-6, draws=10e5):

	grid_filter = np.abs(grid) < 100

	# Get value function parameters
	A0, A1, B0, B1 = Unobservables

	# These are the action specific AR coefficients
	AR0 = np.dot(A0,Theta[1:])
	AR1 = np.dot(A1,Theta[1:])

	# These are the action specific utility constants
	C0 = np.dot(B0[1:], Theta[1:])
	C1 = np.dot(B1[1:], Theta[1:])

	# These are the action specific utility coefficients on the dynamic state
	D0 = B0[0]
	D1 = B1[0]

	# Compute discrete approximation to normal distribution
	jump_density = normal.pdf(grid)

	# Compute the initial value function values
	V_old = np.zeros(grid.shape)
	V_new = np.maximum(
		C0 + np.multiply(D0,grid), 
		C1 + np.multiply(D1,grid))

	distance = np.amax(np.abs(V_old-V_new)[grid_filter])

	while distance > precision:

		V_old = V_new

		# The function "f" maps x to EV(x + e) where e ~ N(0,1) 
		f = interp1d(grid, np.convolve(V_old,jump_density,mode='same'))

		# The continuation values contain EV(Ax + e) for each value in the grid
		CV0 = f(np.multiply(AR0,grid))
		CV1 = f(np.multiply(AR1,grid))

		# Given all this, we can compute the next step of value function iteration
		V_new = np.maximum(
		C0 + np.multiply(D0,grid) + beta*CV0, 
		C1 + np.multiply(D1,grid) + beta*CV1)

		distance = np.amax(np.abs(V_old-V_new)[grid_filter])
		print(distance)

	# VF contains our final estimate of the value function
	VF = V_new

	# Policy contains the optimal action for each value in the grid
	Policy = C0 + np.multiply(D0,grid) + beta*CV0 < np.multiply(D1,grid) + beta*CV1
	
	return {"value_function": VF, "policy": policy, "grid": grid}
