#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import tf2_ros

class ReferenceFramePublisher(Node):
    def __init__(self):
        super().__init__('reference_frame_publisher')
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        # Set the rate at which to publish the reference frame
        self.rate = self.create_rate(10)  # Adjust the rate as needed
        self.timer = self.create_timer(0.5, self.publish_reference_frame)

    def publish_reference_frame(self):
            reference_frame = TransformStamped()
            reference_frame.header.stamp = self.get_clock().now().to_msg()
            reference_frame.header.frame_id = "map"  # Parent frame
            reference_frame.child_frame_id = "reference_frame"  # Child frame
            reference_frame.transform.translation.x = 0.0
            reference_frame.transform.translation.y = 0.0
            reference_frame.transform.translation.z = 0.0
            reference_frame.transform.rotation.x = 0.0
            reference_frame.transform.rotation.y = 0.0
            reference_frame.transform.rotation.z = 0.0
            reference_frame.transform.rotation.w = 1.0
            # Publish the reference frame
            self.tf_broadcaster.sendTransform(reference_frame)


def main(args=None):
    rclpy.init(args=args)
    reference_frame_publisher = ReferenceFramePublisher()
    rclpy.spin(reference_frame_publisher)

    reference_frame_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
