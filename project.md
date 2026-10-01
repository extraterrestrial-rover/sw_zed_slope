# ZED2i Plane Visualizer (AR Overlay)

## Overview
This project is a ROS2 node that overlays a dynamically computed 3D plane onto a live RGB camera feed from a **Stereolabs ZED2i** camera. It is designed for a rover to visualize the ground slope directly ahead of it in real-time. 

Instead of showing a raw 3D point cloud, it acts as an Augmented Reality (AR) Heads-Up Display (HUD) using OpenCV. It computes a plane from the point cloud ahead of the rover and draws a semi-transparent colored plane, the sampled points, and the slope/normal vector data directly onto the live RGB image.

## ROS2 Architecture

**Node Name:** `plane_visualizer`

### Subscribed Topics
- **`/zed/zed_node/rgb/color/rect/image`** (`sensor_msgs/Image`): 
  The rectified left RGB camera feed. Decoded into a BGR numpy array for OpenCV rendering.
- **`/zed/zed_node/rgb/color/rect/camera_info`** (`sensor_msgs/CameraInfo`): 
  Provides the camera intrinsics (K matrix) required to project 3D coordinate points back into the 2D image plane.
- **`/zed/zed_node/point_cloud/cloud_registered`** (`sensor_msgs/PointCloud2`): 
  The dense 3D point cloud aligned with the left RGB camera.

## Core Logic & Technical Implementation

### 1. Fast Point Cloud Parsing
The standard `sensor_msgs_py.point_cloud2` parser can be slow and struggles with certain structured array conversions in newer NumPy versions. This script bypasses it by reading the raw byte buffer using `np.frombuffer(msg.data, dtype=np.float32)`, restructuring it based on `point_step`, and isolating the X, Y, Z float coordinates for extremely fast processing.

### 2. Coordinate Frames & Transformations
The ZED wrapper can output point clouds in either the base camera link frame or the optical frame. The code dynamically detects this via the `msg.header.frame_id`:
- **`zed_left_camera_frame` (Standard ROS)**: X is depth (forward), Y is lateral (left), Z is height (up).
- **`optical_frame`**: Z is depth (forward), X is lateral (right), Y is height (down).

Before projection, all 3D points are converted to the standard Optical Frame so they can be multiplied by the Intrinsic K Matrix.

### 3. Grid Sampling & Depth Isolation
To process the dense point cloud efficiently, the algorithm applies a Field of View (FOV) filter (e.g., 0.1m - 3.0m ahead, ±0.75m laterally). It then finds the dominant depth using a histogram and performs a grid sampling (e.g., 10x10 cells) over the region to ensure an even distribution of points and reduce computation.

### 4. Robust Plane Fitting (RANSAC & SVD)
Instead of relying purely on a fixed set of points, the node uses RANSAC (Random Sample Consensus) on the grid-sampled points to find the dominant planar surface robustly.
- It calculates an inlier mask for the points that fit the plane.
- An SVD (Singular Value Decomposition) is performed on the inliers to refine the plane normal and calculate a **planarity** score.
- As a fallback (if RANSAC fails to find enough inliers), it falls back to sampling 3 fixed geometric target locations ahead of the rover.

### 5. Slope Classification
The normal vector of the fitted plane is used to calculate the ground's **pitch** (forward/backward tilt) and **camber** (left/right tilt). 
Based on the steepness (angle from horizontal), the slope is classified into:
- **FLAT** (<= 10 deg)
- **GENTLE SLOPE** (<= 25 deg)
- **STEEP SLOPE** (<= 45 deg)
- **EXTREME SLOPE** (<= 75 deg)
- **WALL / OBSTACLE** (> 75 deg)

### 6. AR Projection & OpenCV Rendering
Using the projection formula $u = f_x \frac{X}{Z} + c_x$ and $v = f_y \frac{Y}{Z} + c_y$, the 3D plane corners and the sampled points are projected onto the 2D image plane.
- `cv2.fillPoly` and `cv2.addWeighted` are used to draw a semi-transparent colored plane on the ground. The color corresponds to the slope severity.
- A HUD (Heads-Up Display) is drawn detailing the normal vector, plane equation, pitch, camber, and planarity.
- A large text readout of the slope classification is displayed, and walls/obstacles trigger a flashing red border around the screen.

## Requirements
- ROS2 Humble
- ZED ROS2 Wrapper (`zed_wrapper`)
- Python Packages: `numpy`, `opencv-python` (cv2), `rclpy`
