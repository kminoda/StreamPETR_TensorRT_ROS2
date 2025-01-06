import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import cv2
from cv_bridge import CvBridge
from tf2_ros import StaticTransformBroadcaster
from nuscenes.nuscenes import NuScenes
import numpy as np
import time
from tf_transformations import quaternion_from_matrix
import argparse

def quaternion_to_rotation_matrix(q):
    """Convert [x, y, z, w] quaternion to 3x3 rotation matrix."""
    x, y, z, w = q
    R = np.eye(3)
    xx = x*x; yy = y*y; zz = z*z; ww = w*w
    xy = x*y; xz = x*z; xw = x*w
    yz = y*z; yw = y*w; zw = z*w

    R[0,0] = ww + xx - yy - zz
    R[0,1] = 2*(xy - zw)
    R[0,2] = 2*(xz + yw)
    R[1,0] = 2*(xy + zw)
    R[1,1] = ww - xx + yy - zz
    R[1,2] = 2*(yz - xw)
    R[2,0] = 2*(xz - yw)
    R[2,1] = 2*(yz + xw)
    R[2,2] = ww - xx - yy + zz
    return R

def make_4x4(rotation_q, translation_xyz):
    """Make a 4x4 matrix from quaternion (x,y,z,w) and translation (x,y,z)."""
    R = quaternion_to_rotation_matrix(rotation_q)
    T = np.eye(4)
    T[:3,:3] = R
    T[0,3]   = translation_xyz[0]
    T[1,3]   = translation_xyz[1]
    T[2,3]   = translation_xyz[2]
    return T

def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
    M = np.eye(4)
    M[:3,:3] = R
    return quaternion_from_matrix(M)  # [x, y, z, w]

class NuScenesPublisher(Node):
    def __init__(self, nuscenes_data_root):
        super().__init__('nuscenes_publisher')

        # Define the desired resize dimensions
        self.resize_width = 640
        self.resize_height = 480

        self.image_publishers = {
            'CAM_FRONT': self.create_publisher(Image, '/sensing/camera/camera0/image_rect_color', 10),
            'CAM_BACK': self.create_publisher(Image, '/sensing/camera/camera1/image_rect_color', 10),
            'CAM_FRONT_LEFT': self.create_publisher(Image, '/sensing/camera/camera2/image_rect_color', 10),
            'CAM_FRONT_RIGHT': self.create_publisher(Image, '/sensing/camera/camera5/image_rect_color', 10),
            'CAM_BACK_LEFT': self.create_publisher(Image, '/sensing/camera/camera3/image_rect_color', 10),
            'CAM_BACK_RIGHT': self.create_publisher(Image, '/sensing/camera/camera4/image_rect_color', 10)
        }
        self.camera_info_publishers = {
            'CAM_FRONT': self.create_publisher(CameraInfo, '/sensing/camera/camera0/camera_info', 10),
            'CAM_BACK': self.create_publisher(CameraInfo, '/sensing/camera/camera1/camera_info', 10),
            'CAM_FRONT_LEFT': self.create_publisher(CameraInfo, '/sensing/camera/camera2/camera_info', 10),
            'CAM_FRONT_RIGHT': self.create_publisher(CameraInfo, '/sensing/camera/camera5/camera_info', 10),
            'CAM_BACK_LEFT': self.create_publisher(CameraInfo, '/sensing/camera/camera3/camera_info', 10),
            'CAM_BACK_RIGHT': self.create_publisher(CameraInfo, '/sensing/camera/camera4/camera_info', 10)
        }
        self.odom_publisher = self.create_publisher(Odometry, '/localization/kinematic_state', 10)
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.bridge = CvBridge()

        # Load nuScenes dataset
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=nuscenes_data_root, verbose=True)
        
        # Publish TFs as static transforms
        self.publish_static_tfs()

        # Stream one sequence
        self.publish_data()

    def publish_static_tfs(self):
        transforms = []
        sample = self.nusc.sample[0]
        cam_tokens = sample['data'].values()
        for cam_token in cam_tokens:
            sensor_data = self.nusc.get('sample_data', cam_token)
            if "CAM" not in sensor_data['channel'] and sensor_data["channel"] != "LIDAR_TOP":
                continue
            print("Processing sensor: ", sensor_data['channel'])
            cs_record = self.nusc.get('calibrated_sensor', sensor_data['calibrated_sensor_token'])

            sensor2ego_rotation = np.array(cs_record['rotation'])
            sensor2ego_translation = np.array(cs_record['translation'])

            # Create static TF
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = 'base_link'
            transform.child_frame_id = sensor_data['channel']

            transform.transform.translation.x = sensor2ego_translation[0]
            transform.transform.translation.y = sensor2ego_translation[1]
            transform.transform.translation.z = sensor2ego_translation[2]

            transform.transform.rotation.w = sensor2ego_rotation[0]
            transform.transform.rotation.x = sensor2ego_rotation[1]
            transform.transform.rotation.y = sensor2ego_rotation[2]
            transform.transform.rotation.z = sensor2ego_rotation[3]

            transforms.append(transform)
        
        # Broadcast all static transforms at once
        self.tf_broadcaster.sendTransform(transforms)
        
    def publish_data(self):
        prev_timestamp_ns = None
        for sample in self.nusc.sample:
            # Publish localization pose as Odometry
            ego_pose_token = sample['data']['CAM_FRONT']
            ego_pose_data = self.nusc.get('ego_pose', ego_pose_token)

            timestamp_ns = ego_pose_data['timestamp']
            
            # Check if the timestamp difference is greater than 1 second
            if prev_timestamp_ns is not None and (timestamp_ns - prev_timestamp_ns) > 1e6:
                self.get_logger().info("Finished!")
                return
            prev_timestamp_ns = timestamp_ns

            odometry_msg = Odometry()
            odometry_msg.header.stamp = rclpy.time.Time(seconds=timestamp_ns / 1e6).to_msg()
            odometry_msg.header.frame_id = 'map'
            odometry_msg.child_frame_id = 'base_link'

            # Set position
            odometry_msg.pose.pose.position.x = ego_pose_data['translation'][0]
            odometry_msg.pose.pose.position.y = ego_pose_data['translation'][1]
            odometry_msg.pose.pose.position.z = ego_pose_data['translation'][2]

            # Set orientation (quaternion) directly from ego_pose_data
            odometry_msg.pose.pose.orientation.x = ego_pose_data['rotation'][0]
            odometry_msg.pose.pose.orientation.y = ego_pose_data['rotation'][1]
            odometry_msg.pose.pose.orientation.z = ego_pose_data['rotation'][2]
            odometry_msg.pose.pose.orientation.w = ego_pose_data['rotation'][3]

            self.odom_publisher.publish(odometry_msg)

            # Publish images and camera info
            for cam_name, cam_token in sample['data'].items():
                if cam_name not in self.image_publishers:
                    continue

                cam_path, _, cam_intrinsic = self.nusc.get_sample_data(cam_token)
                cam_data = self.nusc.get('sample_data', cam_token)
                cam_calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                image = cv2.imread(self.nusc.get_sample_data_path(cam_token))
                
                if image is None:
                    self.get_logger().warn(f"Image not found at path: {self.nusc.get_sample_data_path(cam_token)}. Skipping...")
                    continue

                height, width = image.shape[:2]

                # Convert and publish image with timestamp
                ros_image = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
                ros_time = rclpy.time.Time(seconds=timestamp_ns / 1e6).to_msg()
                ros_image.header.stamp = ros_time
                ros_image.header.frame_id = cam_data['channel']
                self.image_publishers[cam_name].publish(ros_image)

                # Create and publish CameraInfo
                camera_info = CameraInfo()
                camera_info.header.stamp = ros_time
                camera_info.header.frame_id = cam_data['channel']
                camera_info.width = width
                camera_info.height = height
                camera_info.k[0] = cam_calib['camera_intrinsic'][0][0]
                camera_info.k[1] = cam_calib['camera_intrinsic'][0][1]
                camera_info.k[2] = cam_calib['camera_intrinsic'][0][2]
                camera_info.k[3] = cam_calib['camera_intrinsic'][1][0]
                camera_info.k[4] = cam_calib['camera_intrinsic'][1][1]
                camera_info.k[5] = cam_calib['camera_intrinsic'][1][2]
                camera_info.k[6] = cam_calib['camera_intrinsic'][2][0]
                camera_info.k[7] = cam_calib['camera_intrinsic'][2][1]
                camera_info.k[8] = cam_calib['camera_intrinsic'][2][2]
                camera_info.d = [float(d) for d in cam_calib.get('camera_distortion', [])]
                camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                camera_info.p[0] = cam_calib['camera_intrinsic'][0][0]
                camera_info.p[1] = cam_calib['camera_intrinsic'][0][1]
                camera_info.p[2] = cam_calib['camera_intrinsic'][0][2]
                camera_info.p[3] = 0.0
                camera_info.p[4] = cam_calib['camera_intrinsic'][1][0]
                camera_info.p[5] = cam_calib['camera_intrinsic'][1][1]
                camera_info.p[6] = cam_calib['camera_intrinsic'][1][2]
                camera_info.p[7] = 0.0
                camera_info.p[8] = 0.0
                camera_info.p[9] = 0.0
                camera_info.p[10] = 1.0
                camera_info.p[11] = 0.0
                self.camera_info_publishers[cam_name].publish(camera_info)

            time.sleep(0.5)

def main(args=None):
    parser = argparse.ArgumentParser(description="NuScenes ROS 2 Publisher")
    parser.add_argument('--nuscenes_data_root', type=str, required=True,
                        help='Path to the root directory of your nuScenes data.')
    parsed_args = parser.parse_args()

    rclpy.init(args=args)
    node = NuScenesPublisher(nuscenes_data_root=parsed_args.nuscenes_data_root)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()