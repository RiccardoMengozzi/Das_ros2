###################################################################################
                                # Task 1.2 #
###################################################################################
import numpy as np
import utils
import matplotlib.pyplot as plt

def cost_function(dataset, labels, w, phi_map):
    cost = 0.0
    for i in range(len(dataset)):
        z = dataset[i]
        y = labels[i]
        cost += np.log(1 + np.exp(-y * np.dot(w, phi_map(z))))
    return cost


print("\n","\t"*3,"#"*10, " Task 1.2 ", "#"*10)
##############################
 # Parameters
##############################
M_train = 500  # Number of Training Points
M_test = 500   # Number of points to test the classifier
d = 2    # Dimensionality of each point
learning_rate = 0.1 # Learning rate for the logistic regression
max_iter = 2000# Maximum number of iterations for the logistic regression
tolerance = 1e-8 # Tolerance for the logistic regression

# Set only one classifier to True 
ELLIPSE_CLASSIFIER = True
POLY_CLASSIFIER = False
### TODO : Add other classifiers


##############################
 # Generate the dataset , labels and define the feature map
##############################
if ELLIPSE_CLASSIFIER and not POLY_CLASSIFIER:
    feature_map = utils.elliptic_map # select the feature map for the ellipse
    #feature_map = polynomial_map
    ellipse_center = np.array([-0.30,-0.0])  # Set the center of the ellipse
    ellipse_axes_lengths = np.array([1.0, 1.0])  # Set the lengths of the ellipse axes
    data_lower_bound = np.array([-0.9, -0.9]) # Set the lower bound of the dataset
    data_upper_bound = -data_lower_bound
    dataset, labels, real_weights = utils.generate_ellipse_dataset(M_train,ellipse_center, ellipse_axes_lengths, data_lower_bound, data_upper_bound, noise= False)

elif POLY_CLASSIFIER and not ELLIPSE_CLASSIFIER:
    feature_map = utils.polynomial_map# select the feature map for a linear classifier
    real_weights = np.array([1.8, 2.6, 0.4, -0.8, -6.1, -6.7, 0.6]) # Set the weights of the linear classifier
    data_lower_bound = np.array([-1, -1]) # Set the lower bound of the dataset
    data_upper_bound = np.array([1, 1])
    dataset, labels, real_weights = utils.generate_polynomial_dataset(M_train, real_weights, data_lower_bound, data_upper_bound, noise=False)


utils.plot_2d_dataset(dataset, labels) # Plot the dataset

# zero = utils.logistic_function_zero(dataset, labels, phi_map=feature_map)
# zero = zero/np.linalg.norm(zero)
# print("Zero of the function: w=", zero/np.linalg.norm(zero))
##############################
 # Train the classifier
##############################

w, w_list, gradient, gradient_list, cost_evolution = utils.logistic_regression_gradient_method(dataset, labels, evaluate_phi=feature_map, lr=learning_rate, max_iter=max_iter, tolerance=tolerance, progress_bar=True)

norm = np.linalg.norm(w)
real_cost_value = utils.my_logistic_cost(real_weights*norm, dataset, labels, phi_map=feature_map)
print("Cost Value: l(w)=", real_cost_value)
cost_value = utils.my_logistic_cost(w, dataset, labels, phi_map=feature_map)
print("Cost Value: l(w)=", cost_value)

cost_evolution = np.array(cost_evolution)
cost_evolution[cost_evolution <= 1e-10] = 1e-10
print("cost value", cost_evolution[-1]) 

#std = feature_map(np.ones(d)) # Standard deviation of the feature map
# Transform the weights back to the original scale
#print("Weights in the original scale:", w)
print("Weights:", w/np.linalg.norm(w), "Not Scaled Weights:", w)

## Plot the cost evolution

gradient_norm_evolution = np.zeros(len(gradient_list))
for i in range(len(gradient_list)):
    gradient_norm_evolution[i] = np.linalg.norm(gradient_list[i])

w_list = np.array(w_list)
plt.plot(w_list[:, 0])
plt.plot(w_list[:, 1])
plt.plot(w_list[:, 2])
plt.plot(w_list[:, 3])
plt.plot(w_list[:, 4])
plt.legend(["w0", "w1", "w2", "w3", "w4"])
plt.show()


plt.plot(cost_evolution)
plt.yscale("log")
plt.xlabel("Iteration k")
plt.ylabel("Cost Function")
plt.title("Cost Function Evolution")
plt.grid(True)
 #plt.savefig("task1_2_plots/Elliptic Cost Evolution")
plt.show()

plt.plot(gradient_norm_evolution)
plt.yscale("log")
plt.xlabel("Iteration k")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norm Evolution")
plt.grid(True)
#plt.savefig("task1_2_plots/Elliptic Gradient Norm Evolution")
plt.show()




##############################
 # Test the classifier on the training dataset
##############################
classified_point = utils.classify_points(dataset , w, phi_map=feature_map)
num_different_cells = np.sum(classified_point != labels)
print("Number of misclassified points: ", num_different_cells)
utils.plot_2d_decision_boundary(dataset, labels, w, phi_map=feature_map, real_weights=real_weights)



##############################
 # Generate a test dataset
##############################
if ELLIPSE_CLASSIFIER:
    test_dataset, test_labels, real_weights = utils.generate_ellipse_dataset(M_test, ellipse_center, ellipse_axes_lengths, data_lower_bound, data_upper_bound, noise = False)
elif POLY_CLASSIFIER:
    test_dataset, test_labels, real_weights = utils.generate_polynomial_dataset(M_test, real_weights, data_lower_bound, data_upper_bound, noise=False)


utils.plot_2d_dataset(test_dataset, test_labels) # Plot the dataset
##############################
 # Test the classifier on the evaluation dataset
##############################
classified_point = utils.classify_points(test_dataset, w, phi_map=feature_map)
num_different_cells = np.sum(classified_point != test_labels)
print("Number of misclassified points: ", num_different_cells)
markers  = np.ones(len(test_labels))
markers[classified_point != test_labels] = 0
utils.plot_2d_decision_boundary(test_dataset, test_labels, w, phi_map=feature_map, markers=markers, real_weights=real_weights)

# Calculate the number of misclassified points on the training dataset
classified_points_train = utils.classify_points(dataset, w, phi_map=feature_map)
num_misclassified_train = np.sum(classified_points_train != labels)

# Calculate the number of misclassified points on the test dataset
classified_points_test = utils.classify_points(test_dataset, w, phi_map=feature_map)
num_misclassified_test = np.sum(classified_points_test != test_labels)

# Calculate the percentage of misclassified points
percentage_misclassified_train = num_misclassified_train / M_train * 100
percentage_misclassified_test = num_misclassified_test / M_test * 100

# # Generate the LaTeX table
# latex_table = "\\begin{center}\n"
# latex_table += "\\begin{tabular}{|c|c|c|c|}\n"
# latex_table += "\\hline\n"
# latex_table += "Dataset & N\\_Data & Misclassified Points & Percentage \\\\\n"
# latex_table += "\\hline\n"
# latex_table += "Training & {} & {} & {:.2f}\\% \\\\\n".format(M_train,num_misclassified_train, percentage_misclassified_train)
# latex_table += "Test & {} & {} & {:.2f}\\% \\\\\n".format(M_test, num_misclassified_test, percentage_misclassified_test)
# latex_table += "\\hline\n"
# latex_table += "\\end{tabular}"
# latex_table += "\\end{center}"
# print(latex_table)




