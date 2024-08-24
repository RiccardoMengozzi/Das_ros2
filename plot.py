import os
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


def cost_function(z, r, r0 , gamma):
    


    r0 = r0.reshape(1,2)
    n_agent = len(r)
    s = np.zeros((1,2))
    for i in range(n_agent):
        s += 1/n_agent * z[i]

    cost = 0.0
    for i in range(N):
        corridor_penalty = 0.0
        grad_corridor_penalty = np.array([0, 0])
        k= 2
        
        x = z[i,0]
        y = z[i,1]
        if -2.2<= x <= 2.2:
            g = y**2 -0.5*x**2 - 0.3
            dgx = -x
            dgy = 2*y 
            if g > -5e-2:
                    g = -5e-2
            corridor_penalty = -k*np.log(-g)
            grad_corridor_penalty = -k/g*np.array([dgx, dgy])
        #cost +=  gamma*(z[i]-r[i])@(z[i]-r[i].T) + 1*(s - r0)@(s - r0).T + 2*(z[i]-r0)@(z[i]-r0).T
        if corridor_penalty>0:
            print(corridor_penalty)
        cost +=  gamma*(z[i]-r[i])@(z[i]-r[i].T) + 1*(s - z[i])@(s - z[i]).T + corridor_penalty
    return cost

def cvx_cost_function(z, r, r0, gamma):
    r0 = r0.reshape(2,)  # Ensure r0 is a 2-dimensional vector
    s = cp.sum(z, axis=0) / N  # Mean of z
    n_agent = len(r)
    cost = 0.0
    for i in range(n_agent):
        #cost += gamma * cp.sum_squares(z[i] - r[i]) + 1*cp.sum_squares(s - r0) + 2*cp.sum_squares(z[i] - r0)
        cost += gamma * cp.sum_squares(z[i] - r[i]) + cp.sum_squares(s - z[i]) 
    return cost

def gradient_function(z, r, r0, gamma):
    s = np.zeros((2))
    r0 = r0.reshape(2)
    for i in range(len(z)):
        s += 1/N * z[i]
    
    grad_norm = np.zeros((2))
    for i in range(len(z)):
        #grad_norm += 2*gamma*(z[i]-r[i])   + 2*(s - r0) + 2*2*(z[i]-r0)
        grad_norm += (1/len(z) -1)*2*(s - z[i]) + 2*gamma*(z[i]-r[i])
    return np.linalg.norm(grad_norm)

N = 0
num_lines = []
# Legge i file e popola le liste di agenti e target


for filename in os.listdir('./das'):
    if filename.startswith('agent_') and filename.endswith('_state.txt'):
        filename = "./das/" + filename
        N += 1
        with open(filename, 'r') as file:
            num_lines.append( sum(1 for line in file))

k = min( num_lines)

# Inizializza array per agenti e target
agents = np.zeros((N, k, 2))
targets = np.zeros((N, k, 2))

# The file is composed by line with the following format: ID AgentX AgentY TargetX TargetY
# Read the file and populate the arrays
for filename in os.listdir('./das'):
    if filename.startswith('agent_') and filename.endswith('_state.txt'):
        filename = "./das/" + filename
        with open(filename, 'r') as file:
            for i in range(k):
                line = file.readline()

                agent_id, agentX, agentY, targetX, targetY = line.split()
                agent_id = int(agent_id)
                agents[agent_id, i, 0] = float(agentX)
                agents[agent_id, i, 1] = float(agentY)
                targets[agent_id, i, 0] = float(targetX)
                targets[agent_id, i, 1] = float(targetY)



##########################
 #CVXPY SOLUTION FOR COMPARISON
##########################
# Define the optimization variables
r0 = np.array([0, 0])
gamma = 0.5
global_cost_evolution = np.zeros((k))
grad_norm_evolution = np.zeros((k))

for i in range(k):
    global_cost_evolution[i] = cost_function(agents[:,i,:], targets[:,i,:], r0, gamma)
    grad_norm_evolution[i] = gradient_function(agents[:,i,:], targets[:,i,:], r0, gamma)
opt_sol = cp.Variable((N, 2))

# Define the cost function
cost = cvx_cost_function(opt_sol, targets[:,-1,:], r0, gamma)

# Define and solve the optimization problem
objective = cp.Minimize(cost)
problem = cp.Problem(objective)
problem.solve()
# Results
print("Optimal z:")
print(opt_sol.value)

optimal_solution = opt_sol.value    
optimal_cost = cost_function(optimal_solution, targets[:,-1,:], r0, gamma)

global_cost_evolution = global_cost_evolution - optimal_cost
global_cost_evolution[global_cost_evolution < 1e-9] = 1e-9
global_cost_evolution=global_cost_evolution.reshape(k, )

plt.plot(global_cost_evolution)
plt.xlabel("Iteration k")
plt.ylabel("Global Cost")
plt.yscale("log")
plt.grid()
plt.title("Global Cost Evolution")
#plt.savefig("Global Cost Evolution Rviz Corridor0.png")
plt.show()

plt.plot(grad_norm_evolution)
plt.xlabel("Iteration k")
plt.ylabel("Gradient Norm")
plt.yscale("log")
plt.grid()
plt.title("Gradient Evolution")
#plt.savefig("Gradient Evolution Rviz Corridor0.png")
plt.show()


#now i wanto to see an animation of the agents and the targets 
# where the agents are represented by a circle and the targets by a square

fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
# Generate the animation
def update(frame):
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.clear()
    for agent_id in range(N):

        agent_color = plt.cm.get_cmap('tab10')(agent_id)  # Get a unique color for each agent
        target_color = agent_color
        
        # Plot agent as a circle
        ax.scatter(agents[agent_id, frame, 0], agents[agent_id, frame, 1], color=agent_color, marker='o', edgecolors="black")
        
        # Plot target as a square
        ax.scatter(targets[agent_id, frame, 0], targets[agent_id, frame, 1], color=target_color, marker='s', edgecolors="black")
        
        # Plot agent trajectory as a line
        ax.plot(agents[agent_id, :frame+1, 0], agents[agent_id, :frame+1, 1], color=agent_color, alpha=0.2)

        # Add red region where the equation : y**2 - 0.5 * x**2 - 0.5 >= 0 and -2<x<2 is satisfied
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = Y**2 - 0.5 * X**2 - 0.5
        #ax.contourf(X, Y, Z, levels=[0, 1e9], colors='red', alpha=0.5)

        
ani = FuncAnimation(fig, update, frames=k, interval=1)
plt.show()