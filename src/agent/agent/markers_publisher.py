import rclpy
from rclpy.node import Node
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header





class markers_publisher(Node):
    def __init__(self):
        super().__init__('Markers_Publisher',  allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        #Default values for parameters
        self.agent_number = 5
        self.r0 = np.array([10.0, -5.0])
        if self.get_parameter("N").value is not None:
            self.agent_number = self.get_parameter("N").value
        if self.get_parameter("r0").value is not None:
            self.r0 = self.get_parameter("r0").value
        self.r0 = np.array(self.r0)
        self.marker_publisher = self.create_publisher(MarkerArray, 'Markers', 10)
        self.map_publisher = self.create_publisher(OccupancyGrid, 'Map', 10)

        CORRIDOR = True

        self.grid_size = 10.0  # 5 meters
        self.resolution = 0.02  # 2 cm
        self.width = int(self.grid_size / self.resolution)
        self.height = int(self.grid_size / self.resolution)
        self.occupancy_grid = OccupancyGrid()
        self.grid_data = np.full((self.height, self.width), 0)
        if CORRIDOR:
            for i in range(self.height):
                for j in range(self.width):
                    x = j * self.resolution - self.grid_size / 2
                    y = i * self.resolution - self.grid_size / 2
                    if y**2 - 0.5 * x**2 - 0.5 >= 0 and -2<x<2:
                        self.grid_data[i, j] = 100  # Occupied


        self.received_marker = []
        for i in range(self.agent_number):
            self.create_subscription(Marker, f"agent_{i}_marker", self.listener_callback, 10)
            self.create_subscription(Marker, f"intruder_{i}_marker", self.listener_callback, 10)

        self.timer = self.create_timer(0.1, self.timer_callback)

    def listener_callback(self, msg):
        self.received_marker.append(msg) 

    def publish_occupancy_grid(self):
        # Populate header
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'reference_frame'

        self.occupancy_grid.header = header
        
        self.occupancy_grid.info.resolution = self.resolution
        self.occupancy_grid.info.width = self.width
        self.occupancy_grid.info.height = self.height
        self.occupancy_grid.info.origin.position.x = -self.grid_size / 2
        self.occupancy_grid.info.origin.position.y = -self.grid_size / 2
        self.occupancy_grid.info.origin.position.z = 0.0
        self.occupancy_grid.info.origin.orientation.w = 1.0

        # Flatten the grid data and assign to data field
        self.occupancy_grid.data = self.grid_data.flatten().tolist()

        # Publish the message
        self.map_publisher.publish(self.occupancy_grid)
       

    def timer_callback(self):
        msg = MarkerArray()
        id_unique = {}
        existing_agent_ids = {}
        self.publish_occupancy_grid()
        for marker in self.received_marker:
            if 0<= marker.id < self.agent_number:
                if marker.id not in id_unique:
                    id_unique[marker.id] = marker.pose.position

        if len(id_unique) == self.agent_number:
            average = np.zeros(2)
            for id, position in id_unique.items():
                average += 1/self.agent_number*np.array([position.x, position.y])
            baricenter_marker_msg = Marker()
            baricenter_marker_msg.header.frame_id = "reference_frame"
            baricenter_marker_msg.header.stamp = self.get_clock().now().to_msg()
            baricenter_marker_msg.id = -2
            baricenter_marker_msg.type = Marker.CYLINDER
            baricenter_marker_msg.action = Marker.ADD
            baricenter_marker_msg.pose.position.x = average[0]
            baricenter_marker_msg.pose.position.y = average[1]
            baricenter_marker_msg.pose.position.z = 0.0
            baricenter_marker_msg.pose.orientation.w = 1.0
            baricenter_marker_msg.scale.x = 0.1
            baricenter_marker_msg.scale.y = 0.1
            baricenter_marker_msg.scale.z = 0.1
            baricenter_marker_msg.color.a = 1.0
            baricenter_marker_msg.color.r = 0.0
            baricenter_marker_msg.color.g = 0.0
            baricenter_marker_msg.color.b = 0.0
            self.received_marker.append(baricenter_marker_msg)

            r0_marker_msg = Marker()
            r0_marker_msg.header.frame_id = "reference_frame"
            r0_marker_msg.header.stamp = self.get_clock().now().to_msg()
            r0_marker_msg.id = -1
            r0_marker_msg.type = Marker.CYLINDER
            r0_marker_msg.action = Marker.ADD
            r0_marker_msg.pose.position.x = self.r0[0]
            r0_marker_msg.pose.position.y = self.r0[1]
            r0_marker_msg.pose.position.z = 0.0
            r0_marker_msg.pose.orientation.w = 1.0
            r0_marker_msg.scale.x = 0.05
            r0_marker_msg.scale.y = 0.05
            r0_marker_msg.scale.z = 0.2
            r0_marker_msg.color.a = 1.0
            r0_marker_msg.color.r = 1.0
            r0_marker_msg.color.g = 0.0
            r0_marker_msg.color.b = 0.0
            self.received_marker.append(r0_marker_msg)
            
            #existing_agent_ids = {marker.id for marker in self.received_marker}
            existing_agent_ids = set()

            for marker in self.received_marker:
                if marker.id not in existing_agent_ids:
                    existing_agent_ids.add(marker.id)
                    msg.markers.append(marker)
                    if len(msg.markers) == 2*self.agent_number + 2 :
                        break
                
            self.received_marker = []
            self.marker_publisher.publish(msg)
                


def main(args=None):
    rclpy.init(args=args)
    markers_publisher_node = markers_publisher()
    rclpy.spin(markers_publisher_node)
    markers_publisher_node.destroy_node()
    rclpy.shutdown()
