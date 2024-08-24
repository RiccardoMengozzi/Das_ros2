import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from aggregative_optimization.msg import AggregativeOptimization as AggrOpt
import numpy as np
import time


np.set_printoptions(precision=2)

class Agent_(Node):

    def __init__(self):
        super().__init__('agent', allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        self.agent_id = self.get_parameter("id").value
        self.neighbors = self.get_parameter("Nii").value
        Adj = self.get_parameter("Adj").value
        z_init = self.get_parameter("z_init").value
        r0 = self.get_parameter("r0").value
        r_init = self.get_parameter("r_init").value
        self.gamma = self.get_parameter("gamma").value
        self.alpha = self.get_parameter("alpha").value
    

        #Initialize the variables
        self.z = np.array(z_init)
        self.r0 = np.array(r0)
        self.r = np.array(r_init)
        self.s = self.phi(self.z)
        self.v = self.local_cost_function(self.z, self.r, self.r0, self.s, self.gamma)[2]
        self.s_old = self.s.copy()
        self.v_old = self.v.copy()
        self.z_old = self.z.copy()
        self.get_logger().info(f'Agent_{self.agent_id} Neigh: {self.neighbors}, Adj:{Adj}')

        with open(f"agent_{self.agent_id}_state.txt", "w") as f:
            pass
        
        #Define the color of the agent and its intruder randomly for visualization
        self.random_color = np.random.rand(3)
        
        #Create a dictionary to store the received data from the neighbors 
        self.received_data = {j: [] for j in self.neighbors}
    
        #Create a dictionary to store the aij values for all neighbors
        self.aij = {j: float() for j in self.neighbors}
        for i in range(0, len(Adj), 2):
            self.aij[Adj[i]] = Adj[i+1]
        
        #Create a subscription to the topics of the neighbors
        for j in self.neighbors:
            self.create_subscription(AggrOpt, f"/information_agent_{j}", self.listener_callback, 10)

        #Create a publisher to publish the data to the neighbors
        self.publisher_ = self.create_publisher(AggrOpt, f"/information_agent_{self.agent_id}", 10)

        #Create a publisher to publish the markers for visualization
        self.agent_publisher = self.create_publisher(Marker, f"/agent_{self.agent_id}_marker", 10)
        self.intruder_publisher = self.create_publisher(Marker, f"/intruder_{self.agent_id}_marker", 10)

        
        #Create a timer to publish the data to the neighbors
        timer_period = 0.05 # seconds
        self.iteration = 0
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.published_iteration = 0

        #Publish the first iteration, to start the communication
        time.sleep(3)
        self.publish_first_iteration()


    def publish_first_iteration(self):
                first_msg = AggrOpt()
                first_msg.iteration = 0
                first_msg.agent_id = self.agent_id
                first_msg.v.x = self.v[0]
                first_msg.v.y = self.v[1]
                first_msg.s.x = self.s[0]
                first_msg.s.y = self.s[1]
                self.get_logger().info(f"Agent_{self.agent_id} Published data at iteration {first_msg.iteration}")
                self.publisher_.publish(first_msg)
                self.iteration = 1  

    #Define the functions for the optimization problem NB: if phi is modified, also phi_grad must be modified
    def phi(self, z):
        return z
    
    def phi_grad(self, z):
        return np.eye(2)
    
    #Define the local cost function and its gradient wrt Z and S
    def local_cost_function(self, z, r, r0, s ,gamma):

        # cost = gamma*(z-r).T@(z-r) + (s - r0).T@(s - r0) + 2*(z-r0).T@(z-r0)
        # cost_grad1 = 2*gamma*(z-r) + 2*2*(z-r0)
        # cost_grad2 = 2*(s - r0)

        cost = gamma*(z-r).T@(z-r)   + (s - z).T@(s - z) 
        cost_grad1 = 2*gamma*(z-r) -2*(s-z) 
        cost_grad2 =  + 2*(s - z)
        return cost, cost_grad1, cost_grad2
    
    def local_cost_function(self, z, r, r0, s ,gamma):
        
        corridor_penalty = 0.0
        grad_corridor_penalty = np.array([0, 0])
        k= 2
        
        x = z[0]
        y = z[1]
        if -2.2<= x <= 2.2:
            g = y**2 -0.5*x**2 - 0.3
            dgx = -x
            dgy = 2*y 
            if g > -5e-2:
                    g = -5e-2
            corridor_penalty = -k*np.log(-g)
            grad_corridor_penalty = -k/g*np.array([dgx, dgy])
                  
        
        #cost = gamma*(z-r).T@(z-r) + (s - r0).T@(s - r0) + 2*(z-r0).T@(z-r0) + corridor_penalty
        # cost_grad1 = 2*gamma*(z-r) + 2*2*(z-r0) + grad_corridor_penalty
        # cost_grad2 = 2*(s - r0)

        cost = gamma*(z-r).T@(z-r) + (s - z).T@(s - z) + corridor_penalty
        cost_grad1 = 2*gamma*(z-r) -2*(s-z) + grad_corridor_penalty
        cost_grad2 =  + 2*(s - z)
        #self.get_logger().info(f"Agent_{self.agent_id} Cost: {cost}, Cost_grad1: {cost_grad1}, Cost Obstacle: {corridor_penalty} , Grad Obstscle: {grad_corridor_penalty}")
        
            
        return cost, cost_grad1, cost_grad2
    

    #Define the callback function for the subscription
    def listener_callback(self, msg):
        self.received_data[msg.agent_id].append(msg)
        #self.get_logger().info(f"Agent_{self.agent_id} Received data from Agent_{msg.agent_id} at iteration {msg.iteration}")

    def next_iteration(self):
        
        self.update_data()
        for j in self.neighbors:
            self.received_data[j].pop(0)
        msg = AggrOpt()
        msg.iteration = self.iteration
        msg.agent_id = self.agent_id
        msg.v.x = self.v[0]
        msg.v.y = self.v[1]
        msg.s.x = self.s[0]
        msg.s.y = self.s[1]
        self.get_logger().info(f"Agent_{self.agent_id} Published data at iteration {msg.iteration}")
        self.publisher_.publish(msg)
        self.iteration += 1
        
    
    def update_data(self):
        self.z_old = self.z.copy()
        with open(f"agent_{self.agent_id}_state.txt", "a") as f:
            f.write(f"{self.agent_id} {self.z[0]} {self.z[1]} {self.r[0]} {self.r[1]} \n")
        
        self.s_old = self.s.copy()
        self.v_old = self.v.copy()
        #received_v_s = [[j, np.array([self.received_data[j].v.x, self.received_data[j].v.y]), np.array([self.received_data[j].s.x, self.received_data[j].s.y]), self.received_data[j].iteration ] for j in self.neighbors]
        #self.get_logger().info(f"Agent_{self.agent_id} Received Data: {received_v_s}")

        # Update z
        old_cost_grad1 , old_cost_grad2 = self.local_cost_function(self.z_old, self.r, self.r0, self.s_old, self.gamma)[1:]
        grad_phi = self.phi_grad(self.z_old)
        self.z = self.z - self.alpha*(old_cost_grad1 + grad_phi@self.v)
        
        # Update s
        Adj_sj = 0.0
        for j in self.neighbors:
            sj= np.array([self.received_data[j][0].s.x, self.received_data[j][0].s.y])
            Adj_sj += self.aij[j]*sj
        self.s = Adj_sj + self.phi(self.z) - self.phi(self.z_old)
        
        # Update v
        cost_grad2 = self.local_cost_function(self.z, self.r, self.r0, self.s, self.gamma)[2]
        Adj_vj = 0.0
        for j in self.neighbors:
            vj = np.array([self.received_data[j][0].v.x, self.received_data[j][0].v.y])
            Adj_vj += self.aij[j]*vj
        self.v = (Adj_vj + (cost_grad2 - old_cost_grad2 ))
  

    def publish_markers(self):
        
        marker_msg = Marker()
        marker_msg.header.frame_id = "reference_frame"
        marker_msg.header.stamp = self.get_clock().now().to_msg()
        marker_msg.id = self.agent_id
        marker_msg.type = Marker.SPHERE
        marker_msg.action = Marker.ADD
        marker_msg.pose.position.x = float(self.z[0])
        marker_msg.pose.position.y = float(self.z[1])
        marker_msg.pose.position.z = 0.0
        marker_msg.pose.orientation.w = 1.0
        marker_msg.scale.x = 0.15
        marker_msg.scale.y = 0.15
        marker_msg.scale.z = 0.15
        marker_msg.color.a = 1.0
        marker_msg.color.r = self.random_color[0]
        marker_msg.color.g = self.random_color[1]
        marker_msg.color.b = self.random_color[2]
        #marker_msg.lifetime = Duration(nanosec= int (1e9 * 0.5))
        self.agent_publisher.publish(marker_msg)

        marker_msg = Marker()
        marker_msg.header.frame_id = "reference_frame"
        marker_msg.header.stamp = self.get_clock().now().to_msg()
        marker_msg.id = self.agent_id + 3000
        marker_msg.type = Marker.CUBE
        marker_msg.action = Marker.ADD
        marker_msg.pose.position.x = float(self.r[0])
        marker_msg.pose.position.y = float(self.r[1])
        marker_msg.pose.position.z = 0.0
        marker_msg.pose.orientation.w = 1.0
        marker_msg.scale.x = 0.2
        marker_msg.scale.y = 0.2
        marker_msg.scale.z = 0.05
        marker_msg.color.a = 1.0
        marker_msg.color.r = self.random_color[0]
        marker_msg.color.g = self.random_color[1]
        marker_msg.color.b = self.random_color[2]
        #marker_msg.lifetime = Duration(nanosec= int (1e9 * 0.5))
        self.intruder_publisher.publish(marker_msg)

        

    def timer_callback(self):
        
        # Publish markers for rviz2 visualization
        self.publish_markers()

        if all(len(self.received_data[j]) > 0 for j in self.neighbors):
            # received_data = [self.received_data[j][0].iteration for j in self.neighbors]
            # self.get_logger().info(f"Agent{self.agent_id} Received Data {received_data}")
            if all(self.received_data[j][0].iteration == self.iteration -1  for j in self.neighbors):
                    #iterations = [[self.received_data[j][0].agent_id ,self.received_data[j][0].iteration] for j in self.neighbors]
                    #self.get_logger().info(f"Agent_{self.agent_id}  S: {self.s}, S_old: {self.s_old}, V: {self.v}, V_old: {self.v_old}, Z: {self.z}, Z_old: {self.z_old}, Iter: {self.iteration-1} " )
                    #self.published_iteration = self.iteration
                    #self.get_logger().info(f"Agent_{self.agent_id} At iteration {self.iteration} Received Data: {iterations}")
                    self.next_iteration()


                                


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = Agent_()
    #Time sleep to let all the agents to be initialized properly
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()