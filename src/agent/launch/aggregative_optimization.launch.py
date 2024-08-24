from launch import LaunchDescription
from launch_ros.actions import Node
import networkx as nx
import numpy as np
import math

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


N = 5
z_init = []
radius = 0.6
angle_increment = 2 * math.pi / N

z_init = np.random.rand(N, 2)
z_init = np.random.rand(N, 2) * 0.5 + np.array([-3.0, -3.0]) 
z_init = z_init.tolist()


r_init = []
for i in range(N):
    angle = i * angle_increment
    x = radius * np.cos(angle) + 4
    y = radius * np.sin(angle) + 2
    r_init.append([x, y])

# r_init = np.random.rand(N, 2) *4
# r_init = r_init.tolist()

r0 = np.random.rand(2)
r0 = np.array([3.0, -3.0])
r0 = r0.tolist()


gamma = np.ones(N) * 0.5
gamma = gamma.tolist()
alpha = np.ones(N) * 0.01
alpha = alpha.tolist()

G = nx.path_graph(N)
#G = nx.cycle_graph(N)
#G= nx.erdos_renyi_graph(N, 1)
#G = nx.complete_graph(N)
self_loops = [(node, node) for node in G.nodes()]
G.add_edges_from(self_loops)

Adj = nx.to_numpy_array(G) 
Adj = metropolis_hastings_wheight(Adj)  # Make it doubly stochastic
print(Adj)


def generate_launch_description():

    node_list = []

    for i in range(N):
        Adj_value = np.array([])
        for j in list(G.neighbors(i)):
            Adj_value = np.append(Adj_value, [j, Adj[i, j]])

        Adj_value = Adj_value.tolist()
        
        N_ii = list(G.neighbors(i))
        node_list.append(Node(
            package="agent",
            namespace="agent" + str(i),
            executable="agent",
            parameters=[{"id": i,
                         "Nii": N_ii,
                         "Adj": Adj_value,
                         "z_init": z_init[i],
                         "r0": r0,
                         "r_init": r_init[i],
                         "gamma": gamma[i],
                         "alpha": alpha[i]}],
            output="screen",
           
        ))

    node_list.append(Node(
        package="agent",
        executable="static_transform",
        output="screen",
    ))

    node_list.append(Node(
        package="agent",
        executable="markers_publisher",
        parameters=[{"N": N,
                     "r0": r0}],
        output="screen",
    ))

    
    return LaunchDescription(node_list)