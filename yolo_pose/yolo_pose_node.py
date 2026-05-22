#!/usr/bin/env python3
import os
import site
import sys

VENV_SITE = '/home/jetson/venv/user_following/lib/python3.10/site-packages'
TORCHVISION_EGG = os.path.join(
    VENV_SITE,
    'torchvision-0.20.0a0+afc54f7-py3.10-linux-aarch64.egg',
)
if os.path.isdir(VENV_SITE):
    site.addsitedir(VENV_SITE)
for _p in (TORCHVISION_EGG, VENV_SITE):
    if os.path.exists(_p):
        if _p in sys.path:
            sys.path.remove(_p)
        sys.path.insert(0, _p)
sys.path.insert(2, '/home/jetson/.local/lib/python3.10/site-packages')

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from ultralytics import YOLO

from senior_msg.msg import YoloPose


class YoloPoseNode(Node):
    def __init__(self):
        super().__init__('yolo_pose_node')
        self.declare_parameter('yolo_model', '/home/jetson/yolov8n-pose.engine')
        self.declare_parameter('yolo_conf', 0.45)
        self.declare_parameter('use_yolo_tracking', True)
        self.declare_parameter('tracker_config', 'bytetrack.yaml')
        self.declare_parameter('show_display', False)

        self.model_path = self.get_parameter('yolo_model').value
        self.conf_thres = float(self.get_parameter('yolo_conf').value)
        self.use_tracking = bool(self.get_parameter('use_yolo_tracking').value)
        self.tracker_config = self.get_parameter('tracker_config').value
        self.show_display = bool(self.get_parameter('show_display').value)

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path, task='pose')
        self.get_logger().info(f'YOLO pose model loaded: {self.model_path}')

        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=5,
        )
        self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, cam_qos)
        self.pose_pub = self.create_publisher(YoloPose, '/vision/yolo_pose', 10)
        self.annotated_pub = self.create_publisher(Image, '/camera/annotated', 5)

    def destroy_node(self):
        if self.show_display:
            cv2.destroyAllWindows()
        super().destroy_node()

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(f'image conversion failed: {exc}')
            return

        try:
            results = self._run_yolo(frame)
        except Exception as exc:
            self.get_logger().warn(f'YOLO pose inference failed: {exc}')
            self._publish_empty(msg.header, frame.shape[1], frame.shape[0])
            return

        result = results[0] if results else None
        pose_msg = self._build_pose_msg(result, msg.header, frame.shape[1], frame.shape[0])
        self.pose_pub.publish(pose_msg)

        annotated = self._draw_annotated(frame, result)
        self._publish_annotated(annotated, msg.header)

        if self.show_display:
            cv2.imshow('YOLO Pose', annotated)
            cv2.waitKey(1)

    def _run_yolo(self, frame):
        if self.use_tracking:
            return self.model.track(
                frame,
                classes=[0],
                conf=self.conf_thres,
                persist=True,
                tracker=self.tracker_config,
                verbose=False,
            )
        return self.model.predict(
            frame,
            classes=[0],
            conf=self.conf_thres,
            verbose=False,
        )

    def _build_pose_msg(self, result, header, image_width, image_height):
        msg = YoloPose()
        msg.header = header
        msg.image_width = float(image_width)
        msg.image_height = float(image_height)
        msg.track_id = -1
        msg.keypoints = [0.0] * 34
        msg.keypoint_confidences = [0.0] * 17

        if result is None or result.boxes is None or len(result.boxes) == 0:
            msg.detected = False
            return msg

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = (
            result.boxes.conf.cpu().numpy()
            if result.boxes.conf is not None else np.zeros((len(boxes_xyxy),), dtype=np.float32)
        )
        clss = (
            result.boxes.cls.cpu().numpy().astype(np.int32)
            if result.boxes.cls is not None else np.zeros((len(boxes_xyxy),), dtype=np.int32)
        )

        areas = np.maximum(boxes_xyxy[:, 2] - boxes_xyxy[:, 0], 0.0) * np.maximum(
            boxes_xyxy[:, 3] - boxes_xyxy[:, 1], 0.0)
        scores = confs + (areas / max(float(image_width * image_height), 1.0))
        idx = int(np.argmax(scores))

        x1, y1, x2, y2 = [float(v) for v in boxes_xyxy[idx]]
        msg.detected = True
        msg.confidence = float(confs[idx])
        msg.class_id = int(clss[idx])
        msg.x1 = x1
        msg.y1 = y1
        msg.x2 = x2
        msg.y2 = y2
        msg.cx = (x1 + x2) / 2.0
        msg.cy = (y1 + y2) / 2.0

        if getattr(result.boxes, 'id', None) is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(np.int32)
            msg.track_id = int(track_ids[idx])

        if result.keypoints is not None and result.keypoints.xy is not None:
            kpts_xy = result.keypoints.xy[idx].cpu().numpy()
            flat = []
            for point in kpts_xy[:17]:
                flat.extend([float(point[0]), float(point[1])])
            msg.keypoints = (flat + [0.0] * 34)[:34]

            if result.keypoints.conf is not None:
                conf = result.keypoints.conf[idx].cpu().numpy()
                msg.keypoint_confidences = [float(v) for v in conf[:17]]
            else:
                msg.keypoint_confidences = [1.0] * 17

        return msg

    def _publish_empty(self, header, image_width, image_height):
        msg = YoloPose()
        msg.header = header
        msg.detected = False
        msg.track_id = -1
        msg.image_width = float(image_width)
        msg.image_height = float(image_height)
        msg.keypoints = [0.0] * 34
        msg.keypoint_confidences = [0.0] * 17
        self.pose_pub.publish(msg)

    def _draw_annotated(self, frame, result):
        if result is None:
            return frame
        try:
            return result.plot()
        except Exception as exc:
            self.get_logger().warn(
                f'annotated image render failed: {exc}',
                throttle_duration_sec=2.0)
            return frame

    def _publish_annotated(self, frame, header):
        try:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(f'annotated image conversion failed: {exc}')
            return
        msg.header = header
        self.annotated_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
