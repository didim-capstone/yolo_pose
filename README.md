# yolo_pose

ROS 2 Humble package for shared YOLO pose inference.

`yolo_pose` subscribes to `/camera/image_raw`, runs YOLO pose inference once per image, and publishes the selected person result to `/vision/yolo_pose`.

This package lets `person_follower` and `fall_detection` share one YOLO result instead of running separate YOLO models.

```text
camera_publisher_node
  └── /camera/image_raw
        ↓
yolo_pose_node
  └── /vision/yolo_pose
        ├── person_follower_node
        └── fall_detection_node
```

## Features

- Subscribes to shared camera images
- Loads one YOLO pose TensorRT engine
- Runs person pose inference
- Publishes bbox, bbox center, track ID, keypoints, and confidence values
- Publishes an annotated image for debugging

## Node

| Node | Executable | Description |
| --- | --- | --- |
| `yolo_pose_node` | `yolo_pose_node` | Runs YOLO pose inference on `/camera/image_raw` and publishes `/vision/yolo_pose`. |

## Topics

### Subscribed

| Topic | Type | Description |
| --- | --- | --- |
| `/camera/image_raw` | `sensor_msgs/msg/Image` | Raw image from `camera_publisher_node`. |

### Published

| Topic | Type | Description |
| --- | --- | --- |
| `/vision/yolo_pose` | `senior_msg/msg/YoloPose` | Shared selected-person YOLO pose result. |
| `/camera/annotated` | `sensor_msgs/msg/Image` | Debug image with YOLO annotation. |

## Output Message

`/vision/yolo_pose` publishes `senior_msg/msg/YoloPose`.

| Field | Description |
| --- | --- |
| `detected` | Whether a person was detected. |
| `confidence` | YOLO confidence score. |
| `class_id` | YOLO class ID. For person, this is usually `0`. |
| `track_id` | Tracking ID when tracking is available. |
| `x1, y1, x2, y2` | Bounding box coordinates. |
| `cx, cy` | Bounding box center point. |
| `image_width, image_height` | Source image size. |
| `keypoints[34]` | COCO 17 keypoints as x, y pairs. |
| `keypoint_confidences[17]` | Confidence score for each keypoint. |

If no person is detected:

```text
detected=False
bbox values=0
keypoints=0
keypoint_confidences=0
track_id=-1
```

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `yolo_model` | `/home/jetson/yolov8n-pose.engine` | TensorRT YOLO pose engine path. |
| `yolo_conf` | `0.45` | Confidence threshold. |
| `use_yolo_tracking` | `True` | Use YOLO tracking when available. |
| `tracker_config` | `bytetrack.yaml` | Tracker config file or name. |
| `show_display` | `False` | Show OpenCV debug window. |
| `camera_hfov` | `70.0` | Used for target continuity scoring. |
| `target_switch_penalty_deg` | `12.0` | Penalty for switching far from the previous target angle. |
| `target_reacquire_frames` | `30` | Frames to wait before relocking to a new tracked person. |
| `yolo_min_box_height_ratio` | `0.08` | Person-like bbox filter, aligned with `person_follower`. |
| `yolo_min_box_area_ratio` | `0.003` | Person-like bbox filter, aligned with `person_follower`. |
| `yolo_max_box_width_ratio` | `0.80` | Person-like bbox filter, aligned with `person_follower`. |

## Build

```bash
cd /home/jetson/colcon_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select senior_msg yolo_pose
source install/setup.bash
```

## Run

Start the camera publisher first:

```bash
ros2 run camera_publisher camera_publisher_node
```

Then start YOLO pose:

```bash
ros2 run yolo_pose yolo_pose_node
```

## Run With Parameters

```bash
ros2 run yolo_pose yolo_pose_node --ros-args \
  -p yolo_model:=/home/jetson/yolov8n-pose.engine \
  -p yolo_conf:=0.45 \
  -p use_yolo_tracking:=True \
  -p tracker_config:=bytetrack.yaml
```

## Related Packages

| Package | Relation |
| --- | --- |
| `camera_publisher` | Publishes `/camera/image_raw`. |
| `person_follower` | Subscribes `/vision/yolo_pose` and uses bbox for YOLO-LiDAR following. |
| `fall_detection` | Subscribes `/vision/yolo_pose` and uses keypoints for fall detection. |
| `senior_msg` | Provides `YoloPose.msg`. |

## Check

```bash
ros2 topic list
ros2 topic echo /vision/yolo_pose
ros2 topic hz /vision/yolo_pose
ros2 topic hz /camera/image_raw
```
