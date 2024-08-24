###################################################################################
                                # Task 1.3 #
###################################################################################
import numpy as np
import utils
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
    
np.set_printoptions(precision=3)
print("\n","\t"*3,"#"*10, " Task 1.3 ", "#"*10)



def global_cost_function(dataset, labels, w, phi_map):
    z = dataset
    y = labels
    x =  - y * np.dot(w, phi_map(z))
    if x > 0:
        cost = x + np.log(1 + np.exp(-x))
    else:
        cost = np.log(1 + np.exp(x))
    return cost

def sigmoid(z):
        if z > 0: return 1/(1 + np.exp(-z))
        else: return np.exp(z)/(1 + np.exp(z))
    
def w_global_grad_and_cost(dataset ,labels, w, evaluate_phi):
    cost = 0.0
    w_gradient = np.zeros_like(w)
    for i in range(len(dataset)):
        phi = evaluate_phi(dataset[i])
        z = labels[i]*w@phi.T
        sig_z = sigmoid(-z)
        w_gradient += (-labels[i]*phi*sig_z)
        cost += global_cost_function(dataset[i], labels[i], w, evaluate_phi)
    return cost, w_gradient

##############################
 # Parameters for the classifier
##############################
gradient_iter = 2000
M_train = 200 # Number of Training Points
M_test = 200   # Number of points to test the classifier
d = 2    # Dimensionality of each point
learning_rate = 0.1 # Learning rate for the logistic regression
tolerance = 1e-6 # Tolerance for the logistic regression
# Set only one classifier to True
ELLIPSE_CLASSIFIER = True
POLY_CLASSIFIER = False

##############################
 # Parameters Distributed Optimization
##############################
N = 4 # Number of agents
alpha = 0.1 # Step size for gradient tracking

## Select the  undirected graph to test
#G= nx.erdos_renyi_graph(N, 1)
G = nx.path_graph(N)
#G = nx.cycle_graph(N)
#G = nx.star_graph(N-1)

## Adjacency matrix of the graph
Adj = nx.to_numpy_array(G) + np.eye(N) # Add self loops
Adj = utils.metropolis_hastings_wheight(Adj)  # Make it doubly stochastic


##############################
 # Generate the dataset , labels and define the feature map
##############################
if ELLIPSE_CLASSIFIER and not POLY_CLASSIFIER:
    feature_map = utils.elliptic_map # select the feature map for the ellipse
    ellipse_center = np.array([-0.30,-0.0])  # Set the center of the ellipse
    ellipse_axes_lengths = np.array([1.0, 1.0])  # Set the lengths of the ellipse axes
    data_lower_bound = np.array([-0.9, -0.9]) # Set the lower bound of the dataset
    data_upper_bound = -data_lower_bound
    dataset, labels, real_weights = utils.generate_ellipse_dataset  (M_train,ellipse_center, ellipse_axes_lengths, data_lower_bound, data_upper_bound, noise= True)

elif POLY_CLASSIFIER and not ELLIPSE_CLASSIFIER:
    feature_map = utils.polynomial_map# select the feature map for a linear classifier
    real_weights = np.array([1.8, 2.6, 0.4, -0.8, -6.1, -6.7, 0.6]) # Set the weights of the linear classifier
    data_lower_bound = np.array([-1, -1]) # Set the lower bound of the dataset
    data_upper_bound = np.array([1, 1])
    dataset, labels, real_weights = utils.generate_polynomial_dataset(M_train, real_weights, data_lower_bound, data_upper_bound, noise=False)

utils.plot_2d_dataset(dataset, labels) # Plot the dataset

splitted_datasets, splitted_labels = utils.split_dataset(dataset, labels, N, type='random')
utils.plot_agent_datasets(splitted_datasets, splitted_labels)




##############################
    # Train the classifier in a distributed way
##############################
feature_length = len(feature_map(np.random.randn(d))) 
# Define the state of the agents as feature weight, mean of data, std of data [w, meanm std]
z_init = np.empty((N, feature_length ))
w = np.zeros((N, feature_length))
grad_locals = np.zeros((N, feature_length))
for i in range(N):
    z_init[i] = np.random.randn(feature_length)
    grad_locals[i]= utils.logistic_regression_gradient_method(  splitted_datasets[i], splitted_labels[i], initial_w=z_init[i],
                                                                evaluate_phi=feature_map, lr=learning_rate, max_iter=1,
                                                                tolerance=tolerance, progress_bar=False)[2]

z = np.copy(z_init)
s = grad_locals.copy()


global_cost_evolution = np.zeros((N,gradient_iter))
gradient_evolution = np.zeros((N,gradient_iter, feature_length))
gradient_norm_evolution = np.zeros((gradient_iter))
pbar = tqdm(total=gradient_iter , desc="Logistic Regression Gradient Tracking", ncols=150, colour="cyan")
z_list = []

# Run the gradient tracking algorithm
for k in range(gradient_iter):
    pbar.update(1)
    z_old = z.copy()
    old_grad_locals = grad_locals.copy()
    s_old = np.copy(s)
    z_list.append(z_old/np.linalg.norm(z_old))
    # Update z
    for i in range(N):
        # Update z
        z[i] = Adj[i] @ z_old - alpha * s[i]
        # Update Local gradients with the new z
        grad_locals[i] = utils.logistic_regression_gradient_method( splitted_datasets[i], splitted_labels[i], initial_w=z[i],
                                                                    evaluate_phi=feature_map, lr=learning_rate, max_iter=1,
                                                                    tolerance=tolerance, progress_bar= False)[2]
        #update s
        global_cost_evolution[i, k], gradient_evolution[i,k]  = w_global_grad_and_cost(dataset,labels, z[i], feature_map)
        s[i] = Adj[i] @ s_old + ( grad_locals[i] - old_grad_locals[i] )

    gradient = 0.0
    for i in range(N):
        gradient += gradient_evolution[i,k]
    gradient_norm_evolution[k] = np.linalg.norm(gradient)
    if np.linalg.norm(z - z_old) < 1e-3:
        pbar.total = k
        pbar.close()
        k = gradient_iter
        break
    


z_print= np.copy(z)
for i in range(N):
    global_cost_evolution[global_cost_evolution < 1e-10] = 1e-10
    z_print[i] = z[i]/np.linalg.norm(z[i])
    plt.plot(global_cost_evolution[i])

print("Weights: ", z_print)
plt.title("Global Cost Evolution")
plt.xlabel("Iteration k")
plt.ylabel("Global Cost")
plt.legend([f"Agent {i}" for i in range(N)], loc='upper right', fontsize='x-small', prop={'size': 'small'})
plt.grid(True)
plt.yscale("log")
#plt.savefig("task1_3_plots/Elliptic global_cost Noise")
plt.show()

plt.plot(gradient_norm_evolution)
plt.grid(True)
plt.title("Gradient Norm Evolution")
plt.xlabel("Iteration k")
plt.ylabel("Gradient Norm")
plt.yscale("log")
#plt.savefig("task1_3_plots/Elliptic gradient_norm Noise")
plt.show()

dataset = np.concatenate(splitted_datasets, axis=0)
labels = np.concatenate(splitted_labels, axis=0)

utils.plot_agents_decision_boundary(splitted_datasets, splitted_labels, z, phi_map=feature_map, real_weights=real_weights)


z_list = np.array(z_list)
# Calculate the mean of the last state evolution
mean_state_evolution = np.mean(z_list[-1], axis=0)
mean_state_evolution = mean_state_evolution/np.linalg.norm(mean_state_evolution)
print("Mean of the state evolution: ", mean_state_evolution)



# Calculate the error as the difference between the state evolution and the mean state evolution
error = z_list - mean_state_evolution

# Plot the error norm evolution
plt.figure()
for k in range(N):
    plt.plot(np.linalg.norm(error[:, k, :], axis=1))
plt.xlabel('Iteration')
plt.ylabel('Error Norm')
plt.title('Norm of Error between Agent Mean and Agent States')
plt.legend(['Agent '+str(k) for k in range(N)], loc='upper right')
plt.yscale('log')
plt.xscale('log')
plt.grid(True)
#plt.savefig(f"task1_3_plots/Random Split error_norm")
plt.show()






# Generate the LaTeX table
# latex_table = "\\begin{center}\n"
# latex_table += "\\begin{tabular}{|c|c|c|c|c|}\n"
# latex_table += "\\hline\n"
# latex_table += "Agent & Dataset & N\\_Data & Misclassified Points & Percentage \\\\\n"
# latex_table += "\\hline\n"
# for i in range(N):
#     train_classified_point = utils.classify_points(splitted_datasets[i], z[i], phi_map=feature_map)
#     num_misclassified_train = np.sum(train_classified_point != splitted_labels[i])
#     test_classified_point = utils.classify_points(test_dataset, z[i], phi_map=feature_map)
#     num_misclassified_test = np.sum(test_classified_point != test_labels)
#     percentage_misclassified_train = num_misclassified_train / M_train * 100
#     percentage_misclassified_test = num_misclassified_test / M_test * 100
#     latex_table += "\\hline\n"
#     latex_table += "{} & Training & {} & {} & {:.2f}\\% \\\\\n".format(i, len(splitted_labels[i]), num_misclassified_train, percentage_misclassified_train)
#     latex_table += "{} & Test & {} & {} & {:.2f}\\% \\\\\n".format(i, M_test, num_misclassified_test, percentage_misclassified_test)
#     latex_table += "\\hline\n"
# latex_table += "\\end{tabular}"
# latex_table += "\\end{center}"
# print(latex_table)

