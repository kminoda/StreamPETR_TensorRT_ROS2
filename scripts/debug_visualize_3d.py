import rclpy
from rclpy.node import Node

import cv2
import numpy as np

from sensor_msgs.msg import Image, CameraInfo
from autoware_perception_msgs.msg import DetectedObjects
from cv_bridge import CvBridge

import tf2_ros
import math

class BBoxImageOverlayNode(Node):
    def __init__(self):
        super().__init__('bbox_image_overlay_node')

        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers
        self.sub_image = self.create_subscription(
            Image,
            '/sensing/camera/camera0/image_rect_color',
            self.image_callback,
            10
        )
        self.sub_camera_info = self.create_subscription(
            CameraInfo,
            '/sensing/camera/camera0/camera_info',
            self.camera_info_callback,
            10
        )
        self.sub_objects = self.create_subscription(
            DetectedObjects,
            '/stream_petr/output/objects',
            self.objects_callback,
            10
        )

        self.bridge = CvBridge()

        # Camera params
        self.K = None
        self.frame_id_camera = None
        self.distortion_model = None
        self.d_coeffs = None

        # Latest image
        self.current_image = None
        self.current_image_header = None

        # Optional video writer
        self.video_writer = None

    def camera_info_callback(self, msg: CameraInfo):
        # Store camera intrinsics if they are new
        if self.K is None:
            self.K = np.array(msg.k).reshape(3,3)
            self.frame_id_camera = msg.header.frame_id
            self.distortion_model = msg.distortion_model
            self.d_coeffs = msg.d
            self.get_logger().info(f"Received CameraInfo. K=\n{self.K}")

    def image_callback(self, msg: Image):
        # Convert incoming camera image to OpenCV
        self.current_image_header = msg.header
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.current_image = cv_image

        if self.video_writer is None:
            height, width = cv_image.shape[:2]
            fps = 10.0
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter('output_camera0_overlay.mp4', fourcc, fps, (width, height))
            if not self.video_writer.isOpened():
                self.get_logger().warn("VideoWriter failed to open.")

    def objects_callback(self, msg: DetectedObjects):
        if self.current_image is None or self.K is None:
            return

        overlay_img = self.current_image.copy()

        try:
            transform_stamped = self.tf_buffer.lookup_transform(
                self.frame_id_camera,
                'LIDAR_TOP',
                rclpy.time.Time()
            )
        except tf2_ros.LookupException as e:
            self.get_logger().warn(f"TF not found: {str(e)}")
            return
        except tf2_ros.ExtrapolationException as e:
            self.get_logger().warn(f"TF extrapolation error: {str(e)}")
            return

        T_cam_base = self.transform_to_matrix(transform_stamped)
        K = self.K

        for obj in msg.objects:
            corners_3d_base = self.get_3d_bbox_corners(obj)
            corners_3d_cam = []
            for corner in corners_3d_base:
                corner_cam = T_cam_base @ np.array([corner[0], corner[1], corner[2], 1.0])
                corners_3d_cam.append(corner_cam[:3])

            corners_2d = []
            for pt3 in corners_3d_cam:
                if pt3[2] <= 0:
                    corners_2d = []
                    break
                uv = K @ pt3
                u = uv[0] / uv[2]
                v = uv[1] / uv[2]
                corners_2d.append((int(u), int(v)))

            if len(corners_2d) < 8:
                continue

            edges = [
                (0,1), (1,2), (2,3), (3,0),
                (4,5), (5,6), (6,7), (7,4),
                (0,4), (1,5), (2,6), (3,7)
            ]
            for (idx1, idx2) in edges:
                pt1 = corners_2d[idx1]
                pt2 = corners_2d[idx2]
                cv2.line(overlay_img, pt1, pt2, (0,255,0), 2)

        cv2.imshow("Front Camera BBox Overlay", overlay_img)
        cv2.waitKey(1)
        self.video_writer.write(overlay_img)

    def get_3d_bbox_corners(self, obj):
        """
        Compute 8 corners in base_link coordinates.
        """
        px = obj.kinematics.pose_with_covariance.pose.position.x
        py = obj.kinematics.pose_with_covariance.pose.position.y
        pz = obj.kinematics.pose_with_covariance.pose.position.z

        length = obj.shape.dimensions.x
        width  = obj.shape.dimensions.y
        height = obj.shape.dimensions.z

        w_ = obj.kinematics.pose_with_covariance.pose.orientation.w
        z_ = obj.kinematics.pose_with_covariance.pose.orientation.z
        yaw = 2.0 * math.atan2(z_, w_)

        l2 = length / 2.0
        w2 = width / 2.0
        h2 = height / 2.0

        corners_local = np.array([
            [ l2,  w2, -h2],
            [ l2, -w2, -h2],
            [-l2, -w2, -h2],
            [-l2,  w2, -h2],
            [ l2,  w2,  h2],
            [ l2, -w2,  h2],
            [-l2, -w2,  h2],
            [-l2,  w2,  h2]
        ])

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        Rz = np.array([
            [ cos_y, -sin_y, 0],
            [ sin_y,  cos_y, 0],
            [ 0,      0,     1]
        ])

        corners_global = []
        for c in corners_local:
            cg = Rz @ c + np.array([px, py, pz])
            corners_global.append(cg)

        return corners_global

    def transform_to_matrix(self, transform_stamped):
        """
        Convert TransformStamped to a 4x4 matrix.
        """
        t = transform_stamped.transform.translation
        r = transform_stamped.transform.rotation
        tx, ty, tz = t.x, t.y, t.z
        qw, qx, qy, qz = r.w, r.x, r.y, r.z
        R = self.quaternion_to_rotation_matrix(qx, qy, qz, qw)
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = [tx, ty, tz]
        return T

    def quaternion_to_rotation_matrix(self, x, y, z, w):
        """
        Quaternion to 3x3 rotation matrix.
        """
        r11 = 1 - 2*(y**2 + z**2)
        r12 = 2*(x*y - z*w)
        r13 = 2*(x*z + y*w)
        r21 = 2*(x*y + z*w)
        r22 = 1 - 2*(x**2 + z**2)
        r23 = 2*(y*z - x*w)
        r31 = 2*(x*z - y*w)
        r32 = 2*(y*z + x*w)
        r33 = 1 - 2*(x**2 + y**2)

        return np.array([
            [r11, r12, r13],
            [r21, r22, r23],
            [r31, r32, r33]
        ])

    def destroy_node(self):
        if self.video_writer is not None:
            self.video_writer.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BBoxImageOverlayNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
