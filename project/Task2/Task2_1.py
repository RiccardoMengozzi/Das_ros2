import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cvxpy as cp
import os

np.set_printoptions(precision=4)

def cost_function(z, r, r0 , gamma):
    r0 = r0.reshape(1,2)
    n_agent = len(r)
    s = np.zeros((1,2))
    for i in range(n_agent):
        s += 1/n_agent * z[i]

    cost = 0.0
    for i in range(N):
      #cost +=  gamma*(z[i]-r[i])@(z[i]-r[i].T) + 1*(s - r0)@(s - r0).T + 2*(z[i]-r0)@(z[i]-r0).T
      cost +=  gamma*(z[i]-r[i])@(z[i]-r[i].T) + 1*(s - z[i])@(s - z[i]).T
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

def local_cost_function(z, r, r0, s ,gamma):
    r0 = r0.reshape(2,1)
    z = z.reshape(2,1)
    r = r.reshape(2,1)
    s = s.reshape(2,1)

    #cost = gamma*(z-r).T@(z-r)   + (s - r0).T@(s - r0) + 2*(z-r0).T@(z-r0)
    # cost_grad1 = 2*gamma*(z-r) + 2*2*(z-r0)
    # cost_grad2 =  + 2*(s - r0) 

    cost = gamma*(z-r).T@(z-r)   + (s - z).T@(s - z)
    cost_grad1 = 2*gamma*(z-r) -2*(s-z)
    cost_grad2 =  + 2*(s - z)
    
    cost_grad1 = cost_grad1.reshape(2,)
    cost_grad2 = cost_grad2.reshape(2,)
    cost = cost.reshape(1,)   
    return cost, cost_grad1, cost_grad2


def metropolis_hastings_wheight(A : np.ndarray) -> np.ndarray:
    """
            Compute the Metroplis-Hastings weights of an Adjecency Matrix.
            
            Parameters:
                A (N X N array): Adjacency matrix.
            
            Returns:
                (N X N array): Doubly stochastic  Adjecency Matrix.
        """
    n = len(A)
    B = np.zeros_like(A, dtype=float)
    degrees = np.sum(A, axis=1)
    for i in range(n):
        for j in range(n):
            if A[i, j] == 1 and i != j:
                B[i, j] = 1 / (1 + max(degrees[i], degrees[j]))   
        B[i,i] = 1- np.sum(B[i, :])

    return B

save_dir = "task2_1_plots"
N = 5 # Number of agents
#G = nx.path_graph(N)
#G = nx.star_graph(N-1)
G = nx.cycle_graph(N)
#G = nx.erdos_renyi_graph(N, 0)
gamma = 1
alpha = 0.005
alg_iter = 10000
VISU_OPTIMAL = True

Adj = nx.to_numpy_array(G) + np.eye(N) # Add self loops
Adj = metropolis_hastings_wheight(Adj)  # Make it doubly stochastic
print("Adjacency matrix\n", Adj)


z_init = np.random.randn(N, 2) * 4# Initial state of the agents
#z_init = np.zeros((N, 2)) + np.array([-5,5]) # Initial state of the agents
radius = 8
angle_increment = 2 * np.pi / N
r_init = []
for i in range(N):
    angle = i * angle_increment
    x = radius * np.cos(angle) + 0.4
    y = radius * np.sin(angle) - 0.2
    r_init.append([x, y])

r = np.array(r_init) # Random target
#r = np.random.randn(N, 2) *5 + np.array([3, 4]) # Random target
r0 = np.zeros((1, 2)) #+ np.array([2.3,3]) # Random target

z = z_init.copy()
s = z_init.copy()
v = np.zeros((N, 2))

cost_grad1 = np.zeros((N, 2))
cost_grad2 = np.zeros((N, 2))
for i in range(N):
   cost_grad1[i], cost_grad2[i] = local_cost_function(z[i], r[i], r0, s[i], gamma)[1:]
v = cost_grad2.copy()

z_list = []

global_cost_evolution = np.zeros((alg_iter))
grad_norm_evolution = np.zeros((alg_iter))

for k in range(alg_iter):
    z_old = z.copy()
    s_old = s.copy()
    v_old = v.copy()
    old_cost_grad1 = cost_grad1.copy()
    old_cost_grad2 = cost_grad2.copy()

    z = z_old - alpha*( old_cost_grad1 + v) 
    s = Adj@s_old + z - z_old
    for i in range(N):
        cost_grad1[i], cost_grad2[i] = local_cost_function(z[i], r[i], r0, s[i], gamma)[1:]

    global_cost_evolution[k] = cost_function(z, r, r0, gamma)
    grad_norm_evolution[k] = gradient_function(z, r, r0, gamma)
    v = Adj@v_old + cost_grad2 - old_cost_grad2
    z_list.append(z.copy())

print("Agent z\n", z)

##########################
 #CVXPY SOLUTION FOR COMPARISON
##########################
# Define the optimization variables
opt_sol = cp.Variable((N, 2))

# Define the cost function
cost = cvx_cost_function(opt_sol, r, r0, gamma)

# Define and solve the optimization problem
objective = cp.Minimize(cost)
problem = cp.Problem(objective)
problem.solve()
# Results
print("Optimal z:")
print(opt_sol.value)

optimal_solution = opt_sol.value    
optimal_cost = cost_function(optimal_solution, r, r0, gamma)

global_cost_evolution = global_cost_evolution - optimal_cost
global_cost_evolution[global_cost_evolution < 1e-9] = 1e-9
global_cost_evolution=global_cost_evolution.reshape(alg_iter, )

plt.plot(global_cost_evolution)
plt.xlabel("Iteration k")
plt.ylabel("Global Cost")
plt.yscale("log")
plt.grid()
plt.title("Global Cost Evolution")
#plt.savefig(f"task2_1_plots/Global Cost Evolution Gamma={gamma}, N={N}.png")
plt.show()

plt.show()

optimal_baricenter = np.mean(optimal_solution, axis=0)

# Initialize the plot
fig, ax = plt.subplots()

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
colors = np.arange(N)
color_map = plt.get_cmap('tab20')

agent_colors = color_map(colors / float(N))

scatter_agents = ax.scatter(z_init[:, 0], z_init[:, 1], marker='o', c=agent_colors, edgecolors="black")
scatter_targets = ax.scatter(r[:, 0], r[:, 1], marker='s', c=agent_colors,  s=50, edgecolors="black")
#scatter_r0 = ax.scatter(r0[0, 0], r0[0, 1], marker='x', color='black', zorder=50)
scatter_baricenter = ax.scatter(np.mean(z_init[:, 0]), np.mean(z_init[:, 1]), marker='^', color='red', zorder=0, edgecolors="black")

if VISU_OPTIMAL:
    scatter_optimal_agents = [ax.scatter(optimal_solution[i,0], optimal_solution[i,1], marker="o", color=agent_colors[i], s=100, facecolors='none') for i in range(N)]
    scatter_optimal_barycenter = ax.scatter(optimal_baricenter[0], optimal_baricenter[1], marker='^', color="red",  s=130, facecolors='none')

# Store the trajectory lines
lines = [ax.plot([], [], lw=1, color=agent_colors[i])[0] for i in range(N)]
def update(frame):
    z = z_list[frame]
    baricenter = np.mean(z, axis=0)
    scatter_agents.set_offsets(z)
    scatter_baricenter.set_offsets(baricenter)
    for i in range(N):
        xdata, ydata = lines[i].get_data()
        xdata = np.append(xdata, z[i, 0])
        ydata = np.append(ydata, z[i, 1])
        lines[i].set_data(xdata, ydata)
    if (frame % 15 == 0 and frame<=15*5) or frame==int(alg_iter/2) :  # Save every 10th frame, change this condition as needed
        save_path = os.path.join(save_dir, f'N={N} Gamma={gamma} frame_{frame:04d}.png')
        #plt.savefig(save_path)
    
    return scatter_agents, scatter_baricenter, *lines


ani = FuncAnimation(fig, update, frames=len(z_list)-1, interval=2, blit=True)
if VISU_OPTIMAL:
    plt.legend(["Agent", "Intruder","Agent Baricenter",  "CVXPY Solution"], loc='best', fontsize='x-small', prop={'size': 'small'})
else:
    plt.legend(["Agent", "Intruder",  "Agent Baricenter"], loc='best', fontsize='x-small', prop={'size': 'small'})

plt.show()