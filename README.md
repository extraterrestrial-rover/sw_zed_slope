# ZED2i Plane Visualizer (AR Overlay)

## Overview
This project is a ROS2 (Humble) node that overlays a dynamically computed 3D plane onto a live RGB camera feed from a **Stereolabs ZED2i** camera. It acts as an Augmented Reality (AR) Heads-Up Display (HUD) using OpenCV, designed for a rover to visualize the ground slope directly ahead of it in real-time.

For more technical details about the core logic, coordinate frames, and architecture, refer to [`project.md`](project.md).

## Requirements
- ROS2 Humble
- ZED ROS2 Wrapper (`zed_wrapper`)
- Python Packages: `numpy`, `opencv-python` (cv2), `rclpy`

## Usage

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
