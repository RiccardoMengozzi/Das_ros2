import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import utils



np.set_printoptions(precision=3)

print("\n","\t"*3,"#"*10, " Task 1.1 ", "#"*10)

###################################################################################
                                # Task 1.1 #
###################################################################################

N = 5 # Number of agents
d = 2 # Dimension of the optimization variable
alpha = 0.1 # Step size

## Select the  undirected graph to test
#G= nx.erdos_renyi_graph(N, 0.0)

G = nx.path_graph(N)
graph = "PathGraph"
# G = nx.cycle_graph(N)
# graph = "CycleGraph"
# G = nx.star_graph(N-1)
# graph = "StarGraph"
# G= nx.erdos_renyi_graph(N, 0)
# graph = "ErdosRenyiGraph"


## Adjacency matrix of the graph
Adj = nx.to_numpy_array(G) + np.eye(N) # Add self loops
Adj = utils.metropolis_hastings_wheight(Adj)  # Make it doubly stochastic

print("Adjacency Matrix: \n", Adj)

## Generate the quadratic functions for each agent
Q = [] # Q = [Q0, Q1, ...Q_N-1]
r = [] # r = [r0, r1, ...r_N-1]
z_init = np.zeros((N, d))


for i in range(N):
    np.random.seed(N+i)
    z_init[i] = np.random.randn(d)
    Qi, ri = utils.generate_positive_definite_quadratic(d)
    Q.append(Qi)
    r.append(ri)


# # Prepare LaTeX table code
# latex_table = r"\begin{table}[H]" + "\n"
# latex_table += r"\centering" + "\n"
# latex_table += r"\begin{tabular}{|c|c|c|c|}" + "\n"
# latex_table += r"\hline" + "\n"
# latex_table += r"Agent & Q & r & z\_init \\" + "\n"
# latex_table += r"\hline" + "\n"
# for i in range(N):
#     Qi_str = r"\begin{bmatrix}" + " & ".join([f"{x:.3f}" for x in Q[i].flatten()]) + r"\end{bmatrix}"
#     ri_str = r"\begin{bmatrix}" + " & ".join([f"{x:.3f}" for x in r[i]]) + r"\end{bmatrix}"
#     z_init_str = r"\begin{bmatrix}" + " & ".join([f"{x:.3f}" for x in z_init[i]]) + r"\end{bmatrix}"
#     latex_table += f"{i} & ${Qi_str}$ & ${ri_str}$ & ${z_init_str}$ \\\\" + "\n"
# latex_table += r"\hline" + "\n"
# latex_table += r"\end{tabular}" + "\n"
# latex_table += r"\caption{Values of $Q$, $r$, and $z\_init$ for each agent}" + "\n"
# latex_table += r"\label{tab:values}" + "\n"
# latex_table += r"\end{table}"

# print(latex_table)
    
## Generate the initial values for the optimization variables

# Run the gradient tracking algorithm

z, z_list, gradient_list = utils.quadratic_gradient_tracking(alpha, Adj, Q, r, z_init, max_iter=10000, tol=1e-10)
print("Gradient Tracking Algorithm Found  the values: [z]=\n", z )

# Compare with the optimal value
function_zero = utils.find_quadratic_function_zero(Q, r)
print("Zero of the function: z=", function_zero, "Cost Value: l(z)=", utils.quadratic_function(function_zero, Q, r))

cost_offset = np.abs(utils.quadratic_function(function_zero, Q, r)) 
print("Cost Offset: ", cost_offset)


###################################################################
                    # Plot the results
###################################################################

global_cost_evolution = np.zeros((N, len(z_list)))
local_cost_evolution = np.zeros((N, len(z_list)))

global_gradient_norm_evolution = np.zeros(len(z_list))
local_gradient_norm_evolution = np.zeros((N, len(z_list)))

for i in range(len(z_list)):
    gradient_sum = np.zeros(d)
    for k in range(N):
        global_cost_evolution[k,i] = utils.quadratic_function(z_list[i][k], Q, r)
        local_cost_evolution[k,i] = utils.local_quadratic_cost(z_list[i][k], Q[k], r[k])
        gradient_sum += gradient_list[i][k]
        local_gradient_norm_evolution[k,i] = np.linalg.norm(gradient_list[i][k]) 
    global_gradient_norm_evolution[i] = np.linalg.norm(gradient_sum)

global_cost_evolution += cost_offset
global_cost_evolution[global_cost_evolution<= 10e-11] = 10e-11

# Plot the cost evolution
plt.figure()
[plt.plot(global_cost_evolution[k,:]) for k in range(N)]
plt.xlabel('Iteration k')
plt.ylabel(r'$ \sum_{i=1}^N \ell_i(z^k)$')
plt.yscale('log')
plt.title('Global Cost Evolution')
plt.legend(['Agent '+str(k) for k in range(N)], loc = 'upper right')
plt.grid(True)
#plt.savefig(f"task1_1_plots/N={N}, {graph} global_cost.png")
plt.show()



# Plot the gradient norm evolution  
plt.plot(global_gradient_norm_evolution, 'r')
plt.xlabel('Iteration k')
plt.ylabel(r'$\|\| \nabla \sum_{i=1}^N \ell_i(z^k) \|\|$')
plt.title(r'Global Gradient Norm ')
plt.grid(True)
plt.yscale('log')
#plt.savefig(f"task1_1_plots/N={N}, {graph} gradient_norm.png")
plt.show()

# Check if consensus is reached
consensus_reached = np.allclose(z_list[-1], np.mean(z_list[-1], axis=0))

# Calculate the norm of the error between function_zero and agent states
error = np.array(z_list) - function_zero

# Plot the error norm evolution
plt.figure()
for k in range(N):
    plt.plot(np.linalg.norm(error[:, k, :], axis=1))
plt.xlabel('Iteration')
plt.ylabel('Error Norm')
plt.title('Norm of Error between Optimal Solution and Agent States')
plt.legend(['Agent '+str(k) for k in range(N)], loc='upper right')
plt.yscale('log')
plt.grid(True)
#plt.savefig(f"task1_1_plots/N={N}, {graph} error_norm.png")
plt.show()