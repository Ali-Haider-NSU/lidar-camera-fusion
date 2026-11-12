# Sketch, not run while making the tutorial.
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class ObstacleNode(Node):
    def __init__(self):
        super().__init__("lidar_obstacles")
        self.sub_ = self.create_subscription(PointCloud2, "points", self.on_cloud,
                                             qos_profile_sensor_data)

    def on_cloud(self, msg):
        pts = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
        # ... crop, ground removal, clustering exactly as in this tutorial ...
        self.get_logger().info(f"{len(pts)} points in frame {msg.header.frame_id}")


def main():
    rclpy.init()
    rclpy.spin(ObstacleNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
