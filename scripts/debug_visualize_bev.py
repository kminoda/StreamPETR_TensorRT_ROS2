import rclpy
from rclpy.node import Node
from autoware_perception_msgs.msg import DetectedObjects
import cv2
import numpy as np

class BBoxBEVNode(Node):
    def __init__(self):
        super().__init__('bbox_bev_node')

        # Create a subscriber
        self.subscription = self.create_subscription(
            DetectedObjects,
            '/stream_petr/output/objects',  # Topic name to subscribe
            # "/perception/object_recognition/detection/pointpainting/validation/objects",
            self.listener_callback,
            10)
        
        # Initialize a video writer with MP4 format
        self.video_writer = cv2.VideoWriter(
            'output_bev.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),  # MP4 codec
            10,  # FPS
            (500, 500)  # Width, Height of the output video
        )

    def listener_callback(self, msg):
        print("Callback called")
        # Create a blank BEV image
        bev_image = np.zeros((500, 500, 3), dtype=np.uint8)
        rate = 5

        for obj in msg.objects:
            # Extract bounding box center and dimensions
            cx = int(obj.kinematics.pose_with_covariance.pose.position.x * rate + 250)
            cy = int(obj.kinematics.pose_with_covariance.pose.position.y * rate + 250)
            w = int(obj.shape.dimensions.x * rate)
            l = int(obj.shape.dimensions.y * rate)
            yaw = np.arctan2(obj.kinematics.pose_with_covariance.pose.orientation.z,
                             obj.kinematics.pose_with_covariance.pose.orientation.w) * 2

            # Calculate the corner points of the bounding box
            corners = np.array([
                [cx - w/2, cy - l/2],
                [cx + w/2, cy - l/2],
                [cx + w/2, cy + l/2],
                [cx - w/2, cy + l/2]
            ])

            # Rotate the corners according to the yaw angle
            rotation_matrix = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)]
            ])
            rotated_corners = np.dot(corners - np.array([cx, cy]), rotation_matrix.T) + np.array([cx, cy])

            # Draw the bounding box on the BEV image
            cv2.polylines(bev_image, [rotated_corners.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Write the frame to the video file
        self.video_writer.write(bev_image)

    def destroy_node(self):
        # Release the video writer
        self.video_writer.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BBoxBEVNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
