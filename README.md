# yolo_pose

Shared YOLO pose inference package for ROS 2 Humble.

It subscribes to `/camera/image_raw`, loads the TensorRT YOLO pose engine once,
runs pose inference once per incoming image, and publishes the selected target
person to `/vision/yolo_pose`.

## Role

```text
camera_publisher_node
  └── /camera/image_raw
        ↓
yolo_pose_node
  ├── /vision/yolo_pose
  └── /camera/annotated
```

`person_follower_node` and `fall_detection_node` both consume
`/vision/yolo_pose`, so YOLO is not run twice.

## Build

```bash
cd /home/jetson/colcon_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select senior_msg yolo_pose
source install/setup.bash
```

## Run

Start the camera first:

```bash
ros2 run camera_publisher camera_publisher_node
```

Then run YOLO pose:

```bash
ros2 run yolo_pose yolo_pose_node
```

Usually both are started by:

```bash
ros2 launch person_follower person_follower.launch.py
```

## Topics

### Subscribed

| Topic | Type | Description |
| --- | --- | --- |
| `/camera/image_raw` | `sensor_msgs/msg/Image` | Raw camera frame. |

### Published

| Topic | Type | Description |
| --- | --- | --- |
| `/vision/yolo_pose` | `senior_msg/msg/YoloPose` | Selected person bbox, track id, keypoints, and keypoint confidences. |
| `/camera/annotated` | `sensor_msgs/msg/Image` | Debug image from YOLO result. |

## Parameters

| Parameter | Default |
| --- | --- |
| `yolo_model` | `/home/jetson/yolov8n-pose.engine` |
| `yolo_conf` | `0.45` |
| `use_yolo_tracking` | `True` |
| `tracker_config` | `bytetrack.yaml` |
| `show_display` | `False` |

Check output:

```bash
ros2 topic echo /vision/yolo_pose
ros2 topic hz /vision/yolo_pose
```
