
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

np.set_printoptions(precision=4)

# phi = [x1, x2, x1^2, x2^2, 1] so that w = [w1, w2, w3, w4, b] and the decision boundary is w1*x1 + w2*x2 + w3*x1^2 + w4*x2^2 + b = 0
def elliptic_map(x):
        """
            Feature map for the elliptic classifier.
            
            Parameters:
                x (np.ndarray): Input point of shape (d, ).
            
            Returns:
                np.ndarray: Feature map of the input point of shape (d+1, ).
        """
        
        return np.array([x[0], x[1], x[0]**2, x[1]**2, 1.0])


# phi = [x1, x2, 1] so that w = [w1, w2, b] and the decision boundary is w1*x1 + w2*x2 + b = 0
def linear_map(x):
        """
            Feature map for the linear classifier.
            
            Parameters:
                x (np.ndarray): Input point of shape (d, ).
            
            Returns:
                np.ndarray: Feature map of the input point of shape (d+1, ).
        """
        return np.array([x[0], x[1], 1.0])

def polynomial_map(x):
    return np.array([x[0], x[1], x[0]**2, x[1]**2,  x[0]**3, x[1]**3,  1])

def normalize_dataset(dataset):
    """
    Normalize the features of a dataset.

    Parameters:
        dataset (np.ndarray): Dataset of shape (M, d).

    Returns:
        np.ndarray: Normalized dataset of shape (M, d).
        np.ndarray: Mean of the dataset shape (d, ).
        np.ndarray: Standard deviation of the dataset shape (d, ).
    """
    mean = np.mean(dataset, axis=0)
    std = np.std(dataset, axis=0)
    return (dataset - mean) / std, mean, std

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

def quadratic_gradient (Q: np.ndarray, r:np.ndarray , z : np.ndarray) -> np.ndarray:
        """
            Compute the gradient of a quadratic function.
            
            Parameters:
                Q (NxN ndarray): Quadratic matrix.
                r (Nxd ndarray): Linear term.
                z (Nxd ndarray): Variable.
            
            Returns:
                (Nxd ndarray): Gradient of the quadratic function.
        """
            # Check dimensions
        if Q.shape[0] != Q.shape[1]:
            raise ValueError("Quadratic matrix must be square")
        if Q.shape[0] != len(r) or Q.shape[0] != len(z):
            raise ValueError("Dimensions of the matrices are not compatible")

        return Q @ z + r

def quadratic_gradient_tracking(alpha : float, Adj : np.ndarray, Q_list : np.ndarray, r_list : np.ndarray, z_init: np.ndarray, max_iter=1000 , tol=1e-6) :
    """
        Gradient Tracking algorithm for consensus optimization.

        Parameters:
            alpha (float): Step size.
            Adj (N x N): Adjacency matrices of the graph.
            Q_list (N x (N x N)): Quadratic matrices for all nodes.
            r_list (N x N x d): Linear terms for all nodes.
            z_init (N x d): Initial variable values for all nodes.
            max_iter (int): Maximum number of iterations.
            tol (float): Convergence tolerance.

        Returns:
            (N x d): Final variable values for all nodes.
            (max_iter x (N x D): Variable values for all nodes at each iteration.
            (max_iter x (N x D): Local gradients for all nodes at each iteration.
    """

    N, d = z_init.shape # Number of nodes and dimension of the optimization variable

    # Initialize variables
    z_list = np.zeros((max_iter, N, d))
    grad_locals_list = np.zeros((max_iter, N, d))

    z = np.copy(z_init)
    grad_locals = np.array([quadratic_gradient(Q, r, z) for (Q, r, z) in zip(Q_list, r_list, z_init)])
    s = grad_locals.copy()

    # Main loop
    pbar = tqdm(total=max_iter , desc="Gradient Tracking", ncols=150, colour="cyan")

    for k in range(max_iter):
        pbar.update(1)

        z_old = z.copy()
        z_list[k] = z.copy()
        old_grad_locals = grad_locals.copy()
        grad_locals_list[k]= grad_locals.copy()
        s_old = np.copy(s)
        
        # Update z
        z = Adj @ z_old - alpha * s_old
        # Update Local gradients with the new z
        grad_locals = np.array([quadratic_gradient(Q, r, z) for (Q, r, z) in zip(Q_list, r_list, z)])
        # Update s
        s = Adj @ s_old + ( grad_locals - old_grad_locals )
        
        # Check convergence
        if np.linalg.norm(z - z_old) < tol :
            pbar.total = pbar.n
            pbar.close()
            z_list[k] = z.copy()
            z_list = z_list[:k]
            print("Convergence reached after", k, "iterations")
            break
    
    return z, z_list, grad_locals_list


def generate_positive_definite_quadratic(d):
    """
        Generate a random positive definite quadratic function.
        
        Parameters:
            d (int): Dimension of the quadratic function.
        
        Returns:
            Q (d x d ndarray): Quadratic matrix.
            r (d ndarray): Linear term.

    """
    # Generate a random lower triangular matrix
    L = np.tril(np.random.uniform(size=(d, d)))
    # Make the matrix positive definite
    Q = L @ L.T
    # Generate a random linear term
    r = np.random.rand(d)
    

    return Q, r


def quadratic_function(z, Q: np.ndarray, r: np.ndarray):
    """
      Compute the value of the total cost function.

      Parameters:
            z (N x d): Optimization variable.
            Q (N x d x d): Quadratic matrices for all nodes.
            r (N x d): Linear terms for all nodes.

    """
    N = len(Q)
    quadratic_function_value = 0
    for i in range(N):
        quadratic_function_value += 0.5 * z.T @ Q[i] @ z + r[i].T @ z 
    return quadratic_function_value

def local_quadratic_cost(z, Q: np.ndarray, r: np.ndarray):
    """
        Compute the value of the local cost function.

        Parameters:
            z (d ndarray): Optimization variable.
            Q (d x d): Quadratic matrix.
            r (d ndarray): Linear term.

        Returns:
            float: Value of the local cost function.
    """
    return 0.5 * z.T @ Q @ z + r.T @ z


def find_quadratic_function_zero(Q: np.ndarray, r :np.ndarray):
    """
        Find the optimal value of the quadratic function.
        
        Parameters:
            Q (N x d x d): Quadratic matrices for all nodes.
            r (N x d): Linear terms for all nodes.
        
        Returns:
            (d ndarray): Optimal value of the quadratic function.
    """
    d = r[0].shape[0]
    # Find the zero of the quadratic function
    initial_guess = np.random.normal(size=d)
    result = minimize(quadratic_function, initial_guess, args=(Q, r))
    if result.success:
        zero = result.x
    else:
        print("Optimization did not converge:", result.message)
    
    return zero

def my_logistic_cost(w, dataset, labels, phi_map):
        cost = 0.0
        for i in range(len(dataset)):
            z = dataset[i]
            y = labels[i]
            cost += np.log(1 + np.exp(-y * np.dot(w, phi_map(z))))
        return cost

def logistic_function_zero(dataset, labels, phi_map):

   
    initial_guess = np.random.normal(size=len(phi_map(dataset[0])))
    result = minimize(my_logistic_cost, initial_guess, args=(dataset, labels, phi_map))
    if result.success:
        zero = result.x
    else:
        print("Optimization did not converge:", result.message)
    
    return zero




def logistic_regression_gradient_method(dataset, labels, evaluate_phi :callable, lr=0.5, max_iter=100,tolerance=1e-4, initial_w=None, progress_bar=True):

    def cost_function(data, label, w, phi_map):
        z = data
        y = label
        x =  - y * np.dot(w, phi_map(z))
        if x > 0:
            cost = x + np.log(1 + np.exp(-x))
        else:
            cost = np.log(1 + np.exp(x))
        return cost
    
    def sigmoid(z):
        if z > 0: return 1/(1 + np.exp(-z))
        else: return np.exp(z)/(1 + np.exp(z))
    
    def w_gradient(dataset ,labels, w):
        cost = 0.0
        w_gradient = np.zeros_like(w)
        for i in range(len(dataset)):
            phi = evaluate_phi(dataset[i])
            z = labels[i]*w@phi.T
            sig_z = sigmoid(-z)
            w_gradient += (-labels[i]*phi*sig_z)
            cost += cost_function(dataset[i], labels[i], w, evaluate_phi)
        return w_gradient, cost
                     

    # Initialize weights
    q = len(evaluate_phi(dataset[0]))
    w_list = []
    gradient_w_list = []
    cost_list = []

    if initial_w is None:
        w = np.random.normal(size=q) 
        w = w/np.linalg.norm(w)
    else:
        w = initial_w
    
    if progress_bar:
        pbar = tqdm(total=max_iter , desc="Logistic Regression Evaluation", ncols=150, colour="cyan")
    
    # Optimization loop
    for _ in range(max_iter):
        if progress_bar:
            pbar.update(1)

        # Compute gradients
        gradient_w = np.zeros_like(w)
        gradient_w, cost = w_gradient(dataset,labels, w)
        prev_w = w.copy()
        
        # Update weights and bias
        w_list.append(prev_w)
        cost_list.append(cost)
        gradient_w_list.append(gradient_w)
        w -= lr* gradient_w
        
        # Convergence criteria
        if np.all(np.abs(w/np.linalg.norm(w) - prev_w/np.linalg.norm(prev_w)) < tolerance ):
            if progress_bar:
                pbar.total = pbar.n
                pbar.close()
            break 

    return w, w_list, gradient_w, gradient_w_list, cost_list


def generate_linear_dataset(n_samples, n_features, data_low_bound, data_high_bound, angular_coefficient, bias, noise=False):
 
    dataset = np.random.uniform(low=data_low_bound, high=data_high_bound, size=(n_samples, n_features))
    weights = np.array([angular_coefficient, -1, bias])
    labels = classify_points(dataset, weights, linear_map)
    if noise:
        #Add some random noise to the dataset
        noise = np.random.normal(0, 0.01, dataset.shape)
        dataset += noise

    return dataset, labels, weights

def generate_ellipse_dataset(M, center, ellipse_axis_lengths, data_low_bound, data_high_bound, noise=False):
    # Generate dataset of random points
    def ellipse_coefficients(x, y, a, b):
        # Compute coefficients
        a /= 2.0
        b /= 2.0 
        
        A = -2*x/(a**2)
        B = -2*y/(b**2)
        C = 1/(a**2)
        D = 1/(b**2)
        E = (x/a)**2 + (y/b)**2 -1

        return np.array([A, B, C, D, E])
    
    
    dataset = np.random.uniform(low=data_low_bound, high=data_high_bound, size=(M, len(center)))
    weight = ellipse_coefficients(center[0], center[1], ellipse_axis_lengths[0], ellipse_axis_lengths[1])
    if classify_points([center], weight, elliptic_map) == -1:   
        weight = -weight   
    labels = classify_points(dataset, weight, elliptic_map)
    print("Real Weights:" ,weight/np.linalg.norm(weight))
    if noise:
        #Add some random noise to the dataset
        noise = np.random.normal(0, 0.02, dataset.shape)
        dataset += noise
    
    return dataset, labels, weight

def generate_polynomial_dataset(M, weights, data_low_bound, data_high_bound, noise=False):
    """
    Generate a dataset for a polynomial classifier.

    Parameters:
        M (int): Number of data points.
        d (int): Degree of the polynomial.
        weights (np.ndarray): Weights of the polynomial classifier.
        data_low_bound (float): Lower bound for the data points.
        data_high_bound (float): Upper bound for the data points.
        noise (bool): Whether to add noise to the dataset.

    Returns:
        np.ndarray: Dataset of shape (M, 2).
        np.ndarray: Labels of shape (M, ).
    """
    # Generate dataset of random points
    dataset = np.random.uniform(low=data_low_bound, high=data_high_bound, size=(M, 2))
    labels = classify_points(dataset, weights, polynomial_map)
    if noise:
        # Add some random noise to the dataset
        noise = np.random.normal(0, 0.01, dataset.shape)
        dataset += noise
    print("Real Weights:" ,weights/np.linalg.norm(weights))
    return dataset, labels, weights/np.linalg.norm(weights)

def classify_points(dataset, w, phi_map: callable):
    """
        Classify a dataset using a linear classifier.
    
    Parameters:
        dataset (np.ndarray): Dataset of shape (M, d).
        w (np.ndarray): Weights of the linear classifier of shape of the feature map.
        phi_map (callable): Feature map function.

    Returns:
        np.ndarray: Labels of the dataset of shape (M, ) 1 or -1.
    """

    if len(w) != len(phi_map(dataset[0])):
        raise ValueError("The number of weights must match the number of features.")

    labels = np.zeros(len(dataset))
    for i in range(len(dataset)):
        phi = phi_map(dataset[i])
        if w@phi.T  > 0:
            labels[i] = 1
        else:
            labels[i] = -1
    return labels

def plot_2d_decision_boundary(dataset, labels, w, phi_map : callable, mean=np.array([0.0, 0.0]), markers=None, real_weights=None):

    def decision_boundary(w, x1_grid, x2_grid, mean):
        # Reshape grid arrays into 1D arrays
        x1_flat = x1_grid.flatten()
        x2_flat = x2_grid.flatten()
        # Compute phi_map for each point on the grid
        phi_values = np.array([phi_map([x1-mean[0], x2-mean[1]]) for x1, x2 in zip(x1_flat, x2_flat)])
        # Compute the decision boundary values using dot product
        db_values = np.dot(phi_values, w)
        
        return db_values.reshape(x1_grid.shape)
    
    # Generate a grid of points
    x1_values = np.linspace(min(dataset[:,0]), max(dataset[:,0]), 100)
    x2_values = np.linspace(min(dataset[:,1]),max(dataset[:,1]), 100)
    x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)

    # Compute the decision boundary values for each point on the grid
    db_values = decision_boundary(w,x1_grid, x2_grid, mean)
    # Plot the ellipse
    class_1 = dataset[labels == 1]
    class_2 = dataset[labels == -1]

    

    plt.scatter(class_1[:, 0], class_1[:, 1], color='blue', marker='o', label='C1', edgecolors='k')
    plt.scatter(class_2[:, 0], class_2[:, 1], color='red', marker='o', label='C2', edgecolors='k')

    plt.xlim(min(dataset[:,0]), max(dataset[:,0]))
    plt.ylim(min(dataset[:,1]), max(dataset[:,1]))
    plt.grid(True)
    plt.contour(x1_grid, x2_grid, db_values, levels=[0], colors='green')
    if real_weights is not None:
        db_values_real = decision_boundary(real_weights,x1_grid, x2_grid, mean)
        plt.contour(x1_grid, x2_grid, db_values_real, levels=[0], colors='black', linestyles='dashed')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Decision Boundary')
    # Add markers for well-classified points if provided
    if markers is not None:
        well_classified = dataset[markers == 1]
        misclassified = dataset[markers == 0]
        #plt.scatter(well_classified[:, 0], well_classified[:, 1], color='green', marker='o', label='Well Classified' ,facecolors='none', edgecolors='green' )
        plt.scatter(misclassified[:, 0], misclassified[:, 1], color='cyan', marker='*', label='Misclassified', s=80)
    plt.grid(True)

    actual_db = [Line2D([0], [0], color='green', lw=2, label='Algorithm Decision Boundary')]
    ideal_db = [Line2D([0], [0], color='black', lw=2, label='Ideal Decision Boundary')]
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend(actual_db)
    handles.extend(ideal_db)

    plt.legend(handles=handles, loc='upper right', prop={'size': 'small'})
    #plt.savefig("task1_2_plots/Elliptic Decision_Boundary")
    plt.show()

def plot_agents_decision_boundary(split_datasets, split_labels, weights, phi_map: callable, mean=np.array([0.0, 0.0]), real_weights=None):
    """
    Plot the decision boundaries for multiple agents' datasets.

    Args:
    - split_datasets: list containing N datasets
    - split_labels: list containing N sets of labels corresponding to the datasets
    - weights: list of N weight vectors, each corresponding to an agent
    - phi_map: callable function that applies the feature transformation
    - mean: array of means for centering the data (default: np.array([0.0, 0.0]))
    """
    N = len(split_datasets)
    
    # Create subplots
    fig, axes = plt.subplots(1, N, figsize=(15, 5))
    
    if N == 1:
        axes = [axes]  # Ensure axes is always a list for consistency
    
    # Determine the global limits for x and y axes
    all_X = np.vstack(split_datasets)
    x_min, x_max = all_X[:, 0].min(), all_X[:, 0].max()
    y_min, y_max = all_X[:, 1].min(), all_X[:, 1].max()
    
    def decision_boundary(w, x1_grid, x2_grid, mean):
        # Reshape grid arrays into 1D arrays
        x1_flat = x1_grid.flatten()
        x2_flat = x2_grid.flatten()
        # Compute phi_map for each point on the grid
        phi_values = np.array([phi_map([x1-mean[0], x2-mean[1]]) for x1, x2 in zip(x1_flat, x2_flat)])
        # Compute the decision boundary values using dot product
        db_values = np.dot(phi_values, w)
        return db_values.reshape(x1_grid.shape)
    
    # Generate a grid of points
    x1_values = np.linspace(x_min, x_max, 100)
    x2_values = np.linspace(y_min, y_max, 100)
    x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)
    
    for i, (ax, X_subset, y_subset, w) in enumerate(zip(axes, split_datasets, split_labels, weights)):
        class1 = X_subset[y_subset == 1]
        class2 = X_subset[y_subset == -1]
        
        ax.scatter(class1[:, 0], class1[:, 1], color='blue', marker='o', label='C1', edgecolors='k')
        ax.scatter(class2[:, 0], class2[:, 1], color='red', marker='o', label='C2', edgecolors='k')
        if real_weights is not None:
            db_values_real = decision_boundary(real_weights,x1_grid, x2_grid, mean)
            ax.contour(x1_grid, x2_grid, db_values_real, levels=[0], colors='black', linestyles='dashed')
        # Compute the decision boundary values for each point on the grid
        db_values = decision_boundary(w, x1_grid, x2_grid, mean)
        ax.contour(x1_grid, x2_grid, db_values, levels=[0], colors='green')
        
        # Set equal limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True)
        ax.legend(loc='upper right', prop={'size': 'small'})
        ax.set_title(f'Agent {i + 1} Decision Boundary')
    
    plt.tight_layout()
    #plt.savefig("task1_3_plots/Poly Decision_Boundary.png")
    plt.show()

def plot_2d_dataset(dataset, labels):
    
    class_1 = dataset[labels == 1]
    class_2 = dataset[labels == -1]

    plt.scatter(class_1[:, 0], class_1[:, 1], color='blue', marker='o', label='C1', edgecolors='k')
    plt.scatter(class_2[:, 0], class_2[:, 1], color='red', marker='o', label='C2', edgecolors='k')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Dataset with Binary Labels')
    plt.legend(loc='upper right')
    plt.grid(True)
    #plt.savefig("task1_3_plots/Elliptic Dataset")
    plt.show()



def plot_agent_datasets(split_datasets, split_labels):
    """
    Plot all the agent datasets with different colors.
    Data points with label -1 are plotted as triangles and data points with label 1 are plotted as circles.

    Args:
    - split_datasets: list containing Num_Agent datasets
    - split_labels: list containing Num_Agent sets of labels corresponding to the datasets
    """
    N = len(split_datasets)
    
    # Create subplots
    fig, axes = plt.subplots(1, N, figsize=(15, 5))
    
    if N == 1:
        axes = [axes]  # Ensure axes is always a list for consistency
    
    # Determine the global limits for x and y axes
    all_X = np.vstack(split_datasets)
    x_min, x_max = all_X[:, 0].min()-0.15, all_X[:, 0].max()+0.15
    y_min, y_max = all_X[:, 1].min()-0.15, all_X[:, 1].max()+0.15
    
    for i, (ax, X_subset, y_subset) in enumerate(zip(axes, split_datasets, split_labels)):
        class1 = X_subset[y_subset == 1]
        class2 = X_subset[y_subset == -1]
        
        ax.scatter(class1[:, 0], class1[:, 1], c='red', marker='o', label='C2', edgecolors='k')
        ax.scatter(class2[:, 0], class2[:, 1], c='blue', marker='o', label='C1', edgecolors='k')
        
        # Set equal limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True)
        ax.legend(loc="upper right")
        ax.set_title(f'Agent_{i + 1} Dataset')
   # plt.savefig("task1_3_plots/Elliptic Dataset Random Split")
    plt.tight_layout()
    plt.show()    

def split_dataset(dataset, labels, num_agents, type="column"):
    """
    Randomly split the dataset and labels into num_agents sets while maintaining consistency between dataset and labels.

    Args:
    - dataset: numpy array of shape (M, d) containing the dataset
    - labels: numpy array of shape (M, ) containing the corresponding labels
    - num_agents: integer, number of sets to split the dataset into

    Returns:
    - list of length num_agents, each containing a tuple (sub_dataset, sub_labels) representing a split of the dataset
    """
    # Get the number of data points in the dataset
    num_data_points = dataset.shape[0]

    # Generate random indices for shuffling the dataset
    if type == "random":
        random_indices = np.random.permutation(num_data_points)

        # Shuffle the dataset and labels
        shuffled_dataset = dataset[random_indices]
        shuffled_labels = labels[random_indices]

        # Calculate the number of data points in each split
        num_points_per_split = num_data_points // num_agents

        # Split the dataset and labels into num_agents sets
        split_datasets = []
        split_labels = []
        start_index = 0
        for _ in range(num_agents):
            # Get the indices for the current split
            end_index = start_index + num_points_per_split
            if end_index > num_data_points:
                end_index = num_data_points

            # Extract the dataset and labels for the current split
            split_datasets.append(shuffled_dataset[start_index:end_index])
            split_labels.append(shuffled_labels[start_index:end_index])

            # Update the start index for the next split
            start_index = end_index
    elif type== "column":
        sorted_indices = np.argsort(dataset[:, 0])
        sorted_dataset = dataset[sorted_indices]
        sorted_labels = labels[sorted_indices]

        # Determine the x range for each split
        x_min = sorted_dataset[:, 0].min()
        x_range = sorted_dataset[:, 0].max() - x_min
        split_datasets = []
        split_labels = []

        for i in range(num_agents):
            x_start = x_min + i * x_range / num_agents
            x_end = x_min + (i + 1) * x_range / num_agents

            # Select the points within the current range
            mask = (sorted_dataset[:, 0] >= x_start) & (sorted_dataset[:, 0] < x_end)
            split_datasets.append(sorted_dataset[mask])
            split_labels.append(sorted_labels[mask])

        # Ensure the last split includes the max value point
        mask = sorted_dataset[:, 0] == sorted_dataset[:, 0].max()
        split_datasets[-1] = np.vstack((split_datasets[-1], sorted_dataset[mask]))
        split_labels[-1] = np.hstack((split_labels[-1], sorted_labels[mask]))

        
    else:
        raise ValueError("Invalid type. Supported types are 'random' and 'column'.")
        

    return split_datasets, split_labels

