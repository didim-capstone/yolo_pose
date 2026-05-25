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
        self.declare_parameter('camera_hfov', 70.0)
        self.declare_parameter('yolo_min_box_height_ratio', 0.08)
        self.declare_parameter('yolo_min_box_area_ratio', 0.003)
        self.declare_parameter('yolo_min_box_aspect_ratio', 0.0)
        self.declare_parameter('yolo_max_box_aspect_ratio', 999.0)
        self.declare_parameter('yolo_max_box_width_ratio', 0.80)
        self.declare_parameter('target_switch_penalty_deg', 12.0)
        self.declare_parameter('target_reacquire_frames', 30)
        self.declare_parameter('yolo_tracking_max_failures', 5)

        self.model_path = self.get_parameter('yolo_model').value
        self.conf_thres = float(self.get_parameter('yolo_conf').value)
        self.use_tracking = bool(self.get_parameter('use_yolo_tracking').value)
        self.tracker_config = self.get_parameter('tracker_config').value
        self.show_display = bool(self.get_parameter('show_display').value)
        self.hfov = float(self.get_parameter('camera_hfov').value)
        self.yolo_min_box_height_ratio = float(
            self.get_parameter('yolo_min_box_height_ratio').value)
        self.yolo_min_box_area_ratio = float(
            self.get_parameter('yolo_min_box_area_ratio').value)
        self.yolo_min_box_aspect_ratio = float(
            self.get_parameter('yolo_min_box_aspect_ratio').value)
        self.yolo_max_box_aspect_ratio = float(
            self.get_parameter('yolo_max_box_aspect_ratio').value)
        self.yolo_max_box_width_ratio = float(
            self.get_parameter('yolo_max_box_width_ratio').value)
        self.switch_penalty = float(
            self.get_parameter('target_switch_penalty_deg').value)
        self.target_reacquire_frames = int(
            self.get_parameter('target_reacquire_frames').value)
        self.yolo_tracking_max_failures = int(
            self.get_parameter('yolo_tracking_max_failures').value)

        self.target_track_id = None
        self.target_missing_count = 0
        self.last_target_angle = None
        self.yolo_tracking_failure_count = 0

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
            try:
                results = self.model.track(
                    frame,
                    classes=[0],
                    conf=self.conf_thres,
                    persist=True,
                    tracker=self.tracker_config,
                    verbose=False,
                )
                self.yolo_tracking_failure_count = 0
                return results
            except Exception as exc:
                self.yolo_tracking_failure_count += 1
                if self.yolo_tracking_failure_count >= self.yolo_tracking_max_failures:
                    self.get_logger().warn(
                        f'YOLO tracking {self.yolo_tracking_failure_count}회 연속 실패; '
                        f'tracking 비활성화. 마지막 오류: {exc}')
                    self.use_tracking = False
                else:
                    self.get_logger().warn(
                        f'YOLO tracking 실패 ({self.yolo_tracking_failure_count}/'
                        f'{self.yolo_tracking_max_failures}); predict fallback: {exc}',
                        throttle_duration_sec=3.0)
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
        target_idx = self._select_target_index(result, boxes_xyxy, image_width, image_height)
        if target_idx is None:
            msg.detected = False
            return msg

        confs = (
            result.boxes.conf.cpu().numpy()
            if result.boxes.conf is not None else np.zeros((len(boxes_xyxy),), dtype=np.float32)
        )
        clss = (
            result.boxes.cls.cpu().numpy().astype(np.int32)
            if result.boxes.cls is not None else np.zeros((len(boxes_xyxy),), dtype=np.int32)
        )

        idx = int(target_idx)

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

    def _select_target_index(self, result, boxes_raw, img_w, img_h):
        mask = self._get_person_like_box_mask(boxes_raw, img_w, img_h)
        valid_indices = np.where(mask)[0]

        rejected = int(len(boxes_raw) - np.count_nonzero(mask))
        if rejected:
            self.get_logger().info(
                f'reject non-person-like boxes: {rejected}/{len(boxes_raw)}',
                throttle_duration_sec=1.0)

        if len(valid_indices) == 0:
            self._handle_no_person_boxes()
            return None

        boxes = boxes_raw[valid_indices]
        centers = (boxes[:, 0] + boxes[:, 2]) / 2.0
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        angles = ((centers - img_w / 2.0) / (img_w / 2.0)) * (self.hfov / 2.0)

        track_ids = None
        if self.use_tracking and getattr(result.boxes, 'id', None) is not None:
            all_track_ids = result.boxes.id.cpu().numpy().astype(np.int32)
            track_ids = all_track_ids[valid_indices]

        if track_ids is not None:
            if self.target_track_id is None:
                local_idx = int(np.argmax(areas))
                self.target_track_id = int(track_ids[local_idx])
                self.target_missing_count = 0
                self.last_target_angle = float(angles[local_idx])
                self.get_logger().info(f'target locked: track_id={self.target_track_id}')
                return int(valid_indices[local_idx])

            matches = np.where(track_ids == self.target_track_id)[0]
            if len(matches) > 0:
                local_idx = int(matches[0])
                self.target_missing_count = 0
                self.last_target_angle = float(angles[local_idx])
                return int(valid_indices[local_idx])

            self.target_missing_count += 1
            if self.target_missing_count <= self.target_reacquire_frames:
                self.get_logger().info(
                    f'reacquiring track_id={self.target_track_id} '
                    f'({self.target_missing_count}/{self.target_reacquire_frames})',
                    throttle_duration_sec=0.5)
                return None

            self.get_logger().warn(
                f'track_id={self.target_track_id} lost after '
                f'{self.target_reacquire_frames} frames; relocking')
            self._reset_target_tracking()

            local_idx = int(np.argmax(areas))
            self.target_track_id = int(track_ids[local_idx])
            self.target_missing_count = 0
            self.last_target_angle = float(angles[local_idx])
            self.get_logger().info(f'target relocked: track_id={self.target_track_id}')
            return int(valid_indices[local_idx])

        if self.last_target_angle is None:
            local_idx = int(np.argmax(areas))
        else:
            area_score = areas / max(float(np.max(areas)), 1.0)
            angle_cost = np.abs(angles - self.last_target_angle) / self.switch_penalty
            local_idx = int(np.argmax(area_score - angle_cost))

        self.last_target_angle = float(angles[local_idx])
        return int(valid_indices[local_idx])

    def _get_person_like_box_mask(self, boxes, img_w, img_h):
        widths = np.maximum(boxes[:, 2] - boxes[:, 0], 1.0)
        heights = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
        area_ratios = (widths * heights) / max(float(img_w * img_h), 1.0)
        height_ratios = heights / max(float(img_h), 1.0)
        width_ratios = widths / max(float(img_w), 1.0)
        aspect_ratios = heights / widths
        return (
            (height_ratios >= self.yolo_min_box_height_ratio)
            & (area_ratios >= self.yolo_min_box_area_ratio)
            & (aspect_ratios >= self.yolo_min_box_aspect_ratio)
            & (aspect_ratios <= self.yolo_max_box_aspect_ratio)
            & (width_ratios <= self.yolo_max_box_width_ratio)
        )

    def _handle_no_person_boxes(self):
        self.target_missing_count += 1
        if self.target_missing_count > self.target_reacquire_frames:
            self._reset_target_tracking()
            self.last_target_angle = None

    def _reset_target_tracking(self):
        self.target_track_id = None
        self.target_missing_count = 0

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
