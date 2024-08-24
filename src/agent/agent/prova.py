from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.node import Node
import numpy as np


class map_publisher (Node):
    def __init__(self):
        super().__init__('Map_Publisher',  allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.map_publisher = self.create_publisher(OccupancyGrid, 'Map', 10)
        L = 5
        resolution = 0.05
        n = int(L/resolution)
        self.map2d = np.array((n,n))




def local_cost_function(self, z, r, r0, s ,gamma):
        b = 0.1
        a = 1.6
        cost = gamma*(z-r).T@(z-r) + (s - r0).T@(s - r0) - np.log((-z[1] +a/2 +b*z[0]**2))  - np.log((+z[1]+a/2+b*z[0]**2))
        cost_grad1 = 2*gamma*(z-r) + np.array([-2*b*z[0]/(-z[1] +a/2 +b*z[0]**2), 1/(-z[1] +a/2 +b*z[0]**2)]) + np.array([-2*b*z[0]/(z[1]+a/2+b*z[0]**2),  -1/(z[1]+a/2+b*z[0]**2)])
        self.get_logger().info(f"Agent_{self.agent_id} Cost: {cost}, Cost_grad1: {cost_grad1}, Cost Barrier 1: {- 0.01*np.log((-z[1] +a/2 +b*z[0]**2))} , Cost Barrier 2: {- 0.01*np.log((+z[1]+a/2+b*z[0]**2))}")
        cost_grad2 = 2*(s - r0)
        return cost, cost_grad1, cost_grad2
