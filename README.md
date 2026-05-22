# yolo_pose

ROS 2 Humble shared YOLO pose inference package.

This package subscribes to `/camera/image_raw`, loads the YOLO pose engine once,
runs pose inference once per image, and publishes one shared result to
`/vision/yolo_pose`.

It is separated so `person_follower` and `fall_detection` do not each run their
own YOLO model.

## Role In The Current Structure

```text
camera_publisher_node
  └── publishes /camera/image_raw
        ↓
yolo_pose_node
  └── publishes /vision/yolo_pose
        ├── person_follower_node subscribes
        └── fall_detection_node subscribes
```

`scene_description_node` does not use `/vision/yolo_pose`; it subscribes to the
camera image directly for VLM input.

## Node

| Node | Executable | Role |
| --- | --- | --- |
| `yolo_pose_node` | `yolo_pose_node` | Runs YOLO pose once on each camera frame and publishes the selected person result. |

## Subscribed Topic

| Topic | Type | Description |
| --- | --- | --- |
| `/camera/image_raw` | `sensor_msgs/msg/Image` | Raw image from `camera_publisher_node`. |

## Published Topics

| Topic | Type | Description |
| --- | --- | --- |
| `/vision/yolo_pose` | `senior_msg/msg/YoloPose` | Shared selected-person pose result. |
| `/camera/annotated` | `sensor_msgs/msg/Image` | Optional debug image with YOLO annotation. |

## Output Message

`/vision/yolo_pose` contains:

```text
detected
confidence
class_id
track_id
x1, y1, x2, y2
cx, cy
image_width, image_height
keypoints[34]
keypoint_confidences[17]
```

If no person is detected, `detected=False` and bbox/keypoint values are zero.

## Build

```bash
cd /home/jetson/colcon_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select senior_msg yolo_pose
source install/setup.bash
```

## Run This Package

Start the camera stream first:

```bash
ros2 run camera_publisher camera_publisher_node
```

Then start YOLO pose:

```bash
ros2 run yolo_pose yolo_pose_node
```

After `/vision/yolo_pose` is being published, consumers can be started:

```bash
ros2 run person_follower person_follower_node
ros2 run fall_detection fall_detection_node
```

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `yolo_model` | `/home/jetson/yolov8n-pose.engine` | TensorRT YOLO pose engine path. |
| `yolo_conf` | `0.45` | YOLO confidence threshold. |
| `use_yolo_tracking` | `True` | Use YOLO tracking when available. |
| `tracker_config` | `bytetrack.yaml` | Tracker config name/path. |
| `show_display` | `False` | Show OpenCV debug window. |

## Check

```bash
ros2 topic list
ros2 topic echo /vision/yolo_pose
ros2 topic hz /vision/yolo_pose
ros2 topic hz /camera/image_raw
```
