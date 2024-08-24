import numpy as np
import cvxpy as cp

def cost_function(z, r, r0, gamma):
    r0 = r0.reshape(2,)  # Ensure r0 is a 2-dimensional vector
    s = cp.sum(z, axis=0) / N  # Mean of z

    cost = 0.0
    for i in range(N):
        cost += gamma * cp.sum_squares(z[i] - r[i]) + cp.sum_squares(s - r0) + cp.sum_squares(z[i] - s)
    return cost

# Parameters
N = 5
gamma = 1.0
z_init = np.ones((N, 2)) * 5
r0 = np.ones((1, 2)) * 0  # Random target
radius = 5.0
angle_increment = 2 * np.pi / N
r_init = []
for i in range(N):
    angle = i * angle_increment
    x = radius * np.cos(angle) + 0.7
    y = radius * np.sin(angle) - 0.9
    r_init.append([x, y])

r = np.array(r_init)  # Random target

# Define the optimization variables
opt_sol = cp.Variable((N, 2))

# Define the cost function
cost = cost_function(opt_sol, r, r0, gamma)

# Define and solve the optimization problem
objective = cp.Minimize(cost)
problem = cp.Problem(objective)
problem.solve()

# Results
print("Optimal z:")
print(opt_sol.value)
print("Optimal cost:")
print(problem.value)
