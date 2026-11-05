# ZED2i Plane Visualizer (AR Overlay)

## Overview
This project is a ROS2 node that overlays a dynamically computed 3D plane onto a live RGB camera feed from a **Stereolabs ZED2i** camera. It acts as an Augmented Reality (AR) Heads-Up Display (HUD) using OpenCV, designed for a rover to visualize the ground slope directly ahead of it in real-time. It also scans for pits, trenches, drop-offs and obstacles ahead of the rover.

For more technical details about the core logic, coordinate frames, and architecture, refer to [`project.md`](project.md).

## Requirements
- ROS2 Humble
- ZED ROS2 Wrapper (`zed_wrapper`)
- Python Packages: `numpy`, `opencv-python` (cv2), `rclpy`
- (Optional, only for `--sdk` mode) ZED SDK with `pyzed`

## Usage

### 1. Local mode (display on the machine running the script)

1. **Start the ZED camera node:**
   Make sure the ZED ROS2 wrapper is publishing data:
   ```bash
   ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i
   ```

2. **Run the visualizer:**
   You can use the provided script to start the node:
   ```bash
   ./run.sh
   ```
   Or run the Python script directly:
   ```bash
   python3 plane_visualizer.py
   ```

### 2. Remote mode (laptop on the rover, viewing from the base station)

The laptop inside the rover runs the node with `--headless` (no local window).
The rendered overlay is published as a ROS2 topic, so the base station can view it
without `ssh -X` (X11 forwarding is laggy). Both machines must use the same `ROS_DOMAIN_ID`.

**On the base station, SSH into the rover laptop:**
```bash
ssh <user>@<rover-laptop-ip>
```

**Rover terminal 1 (ZED driver):**
```bash
source /opt/ros/humble/setup.bash
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i
```

**Rover terminal 2 (second SSH session or `tmux`):**
```bash
source /opt/ros/humble/setup.bash
python3 plane_visualizer.py --headless
```

**Base station (local terminal, not the SSH one):**
```bash
source /opt/ros/humble/setup.bash
ros2 run rqt_image_view rqt_image_view /slope_visualizer/overlay
```

**If `rqt_image_view` stays blank** (the overlay is published with BEST_EFFORT QoS), use:
```bash
ros2 run image_tools showimage --ros-args -r image:=/slope_visualizer/overlay -p reliability:=best_effort
```

**Check that frames are arriving:**
```bash
ros2 topic list | grep -E "zed|slope"
ros2 topic hz /slope_visualizer/overlay
```

### 3. ZED SDK mode (optional)

Reads directly from the ZED SDK instead of ROS topics. **Stop the ZED ROS wrapper first**,
because only one process can open the camera.
```bash
python3 plane_visualizer.py --sdk            # local window
python3 plane_visualizer.py --sdk --headless # publish overlay only
```
In a local window: `1`/`2` exposure -/+, `3`/`4` gain -/+, `0` auto exposure on/off.

### Command-line flags

| Flag | Meaning |
|------|---------|
| `--headless` | No OpenCV window; publish the overlay on `/slope_visualizer/overlay` |
| `--sdk` | Grab image and point cloud straight from the ZED SDK (stop the ROS wrapper first) |

### Keyboard (local window only)

| Key | Action |
|-----|--------|
| `Q` / `Esc` | Quit |
| `D` | Print the per-lane hazard profile (debug) |

## ROS2 Topics

### Published (sent to the base station)

| Topic | Type | Notes |
|-------|------|-------|
| `/slope_visualizer/overlay` | `sensor_msgs/Image` | Rendered AR overlay, `bgr8`, 960x540 (`PUB_RESIZE`), QoS: BEST_EFFORT, VOLATILE, depth 1 |

This is the only topic the node publishes. The slope angle, classification and hazard
warnings are drawn into the image, not published as separate messages.

### Subscribed (ROS mode, from the ZED wrapper)

| Topic | Type |
|-------|------|
| `/zed/zed_node/point_cloud/cloud_registered` | `sensor_msgs/PointCloud2` |
| `/zed/zed_node/rgb/color/rect/image` | `sensor_msgs/Image` |
| `/zed/zed_node/rgb/color/rect/camera_info` | `sensor_msgs/CameraInfo` |

Topic names can differ between ZED wrapper versions. Confirm with `ros2 topic list`.

## Configuration

Key parameters are at the top of `plane_visualizer.py`:
- `CAM_HEIGHT`: measured height of the ZED left camera above the floor (metres). **Set this accurately.**
- `FOV_DEPTH_MIN` / `FOV_DEPTH_MAX` / `FOV_LAT_HALF`: region ahead used for slope fitting.
- `PUB_RESIZE`: overlay resolution published to the base (reduce for bandwidth).
- `HAZ_*`: pit, drop-off and obstacle detection thresholds.