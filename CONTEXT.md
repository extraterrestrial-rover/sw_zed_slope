# ZED2i Slope Visualizer — Project Context for AI Agents

## What This Project Does

This is a **ROS2 (Humble) Python node** that runs alongside a Stereolabs **ZED2i** stereo camera on a rover. Its purpose is to detect the slope of the terrain ahead of the rover and display it as a real-time AR overlay on the live RGB camera feed. It is designed to be viewed from a **base station** over a video stream so a remote driver can judge whether the terrain is safe to drive on.

The key outputs visible on screen are:
- A **coloured semi-transparent plane polygon** projected onto the ground ahead
- **Green/red dots** on the image showing which sampled 3D points belong to the detected surface (green = inlier, red = outlier/background)
- A **bottom status bar** showing the signed slope angle (e.g. `+17.4`) and classification label (FLAT / GENTLE SLOPE / STEEP SLOPE / EXTREME SLOPE / PIT / TRENCH / WALL / OBSTACLE) in a colour that matches severity
- A **small compact HUD card** in the top-left corner with normal vector, pitch, camber, planarity, and inlier count

---

## Hardware & Software Setup

- **Camera**: Stereolabs ZED2i stereo-depth camera
- **OS**: Ubuntu 22.04
- **ROS version**: ROS2 Humble
- **Launch command for camera**:
  ```
  ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i
  ```
- **Script to run the visualizer**:
  ```
  cd ~/zed_slope && bash run.sh
  ```
- **Python deps**: `numpy`, `opencv-python`, `rclpy` (comes with ROS2)

---

## ROS2 Topics Subscribed

| Topic | Message Type | Purpose |
|---|---|---|
| `/zed/zed_node/rgb/color/rect/image` | `sensor_msgs/Image` | Live left RGB camera feed |
| `/zed/zed_node/rgb/color/rect/camera_info` | `sensor_msgs/CameraInfo` | Intrinsic K matrix for 3D→2D projection |
| `/zed/zed_node/point_cloud/cloud_registered` | `sensor_msgs/PointCloud2` | Dense 3D point cloud aligned to left camera |

---

## Code Architecture (`plane_visualizer.py`)

### Constants (top of file — all tunable)

```python
FOV_DEPTH_MAX = 3.0     # metres ahead to consider
FOV_LAT_HALF  = 0.75    # +/- metres lateral (1.5m wide corridor)
GRID_M = 10, GRID_N = 10    # grid for spatial sampling
RANSAC_ITERATIONS = 60      # RANSAC tries
RANSAC_THRESH = 0.03        # inlier threshold (metres)
MIN_INLIERS = 20            # minimum inliers for a valid plane
DEPTH_BIN_WIDTH = 0.15      # depth histogram bin width (metres)
NORMAL_BUF_SIZE = 10        # temporal smoothing frames
THR_FLAT=10, THR_GENTLE=25, THR_STEEP=45, THR_EXTREME=75  # degrees
MIN_PLANARITY = 0.50        # below this = uncertain
MIN_INLIER_RATIO = 0.25     # below this = uncertain
```

### Key Functions

**`grid_sample_fov(pts, depth_col, lat_col)`**
Divides the FOV-filtered point cloud into a 10x10 grid (depth × lateral). Takes the median-depth point from each non-empty cell, yielding ~100 spatially spread points regardless of cloud density.

**`fit_plane_svd(pts, depth_col)`**
Fits a plane to N points via SVD (Principal Component Analysis). The last row of Vt is the best-fit normal. Normal is oriented to point UP (using the `up_col` axis for the frame). Returns: `n, d, centroid, planarity_score` where `planarity = 1 - S[2]/S[0]`.

**`ransac_plane(pts, depth_col)`**
Wraps fit_plane_svd with 60-iteration RANSAC. Randomly samples 3 points per iteration, counts inliers within 3cm, keeps best set, refits SVD on those inliers. Returns None if fewer than 20 inliers found.

**`find_dominant_depth(pts, depth_col)`**
Builds a 1D histogram of depth values and masks only points near the dominant depth bin. Used to isolate the nearest flat surface and discard walls/background before RANSAC.

**`compute_slope(n, depth_col)`**
Computes the **signed slope angle** from the smoothed plane normal:
- **Positive angle** = terrain tilts upward ahead (hill, ramp)
- **Negative angle** = terrain tilts downward ahead (pit, trench, depression)

Sign is determined by the depth component of n (after orienting n to face up):
- `n[depth_col] < 0` → normal leans backward → upslope → positive
- `n[depth_col] > 0` → normal leans forward → downslope → negative

Classification labels:
| |Upward (+)|Downward (−)|
|---|---|---|
|0–10°|FLAT|FLAT (DIP)|
|10–25°|GENTLE SLOPE|GENTLE DESCENT|
|25–45°|STEEP SLOPE|STEEP DESCENT|
|45–75°|EXTREME SLOPE|PIT / TRENCH|
|75–90°|WALL / OBSTACLE|WALL / OBSTACLE|

**`camera_to_optical(pts)`**
Converts from `zed_left_camera_frame` (X=forward, Y=left, Z=up) to optical frame (X=right, Y=down, Z=forward) for correct projection through the K matrix.

**`project_pts(pts_cam, K)`**
Projects 3D points in `zed_left_camera_frame` to 2D pixel coords using `camera_to_optical` then the standard pinhole projection formula.

**`plane_corners_3d(pts3, n, half)`**
Builds 4 corners of a square plane centred on the point cluster centroid, using tangent vectors in the plane.

**`three_fixed_points(pts, lat_col)`** ← **FALLBACK ONLY**
Used only when RANSAC fails. Picks 3 points at fixed equilateral-triangle positions relative to the rover: P1 at 0.5m ahead (centre), P2 at 1.0m ahead-left, P3 at 1.0m ahead-right. Averages nearby point cloud points for stability.

### Processing Pipeline (`_cloud_cb`)

```
Raw PointCloud2 (msg.data)
  │
  ▼ np.frombuffer → float32 → reshape → extract XYZ → filter NaN/Inf
  │
  ▼ Detect frame_id ("optical" vs "zed_left_camera_frame")
    → depth_col=2, lat_col=0  (optical)
    → depth_col=0, lat_col=1  (zed_left)
  │
  ▼ FOV filter: depth 0.1–3.0m, lateral +/-0.75m
  │
  ▼ find_dominant_depth()  → isolate nearest surface
  │
  ▼ grid_sample_fov()      → ~100 spatially spread points
  │
  ▼ ransac_plane()         → n, d, centroid, planarity, inlier_mask
    (falls back to three_fixed_points if RANSAC returns None)
  │
  ▼ Normal smoothing buffer (deque maxlen=10) → smoothed_n
  │
  ▼ compute_slope(smoothed_n) → signed angle, classification, colour
  │
  ▼ plane_corners_3d()    → 4 corners of visible plane quad
  │
  ▼ Store in self._latest dict → renderer picks it up
```

### Coordinate Frame Notes

The ZED wrapper publishes the point cloud in `zed_left_camera_frame`:
- **X** = forward (depth)
- **Y** = lateral left
- **Z** = up

Before projecting with the K matrix, points are converted to optical frame:
- optical_X = -camera_Y
- optical_Y = -camera_Z
- optical_Z = camera_X

### Rendering (`draw_overlay`)

1. Draws semi-transparent filled plane polygon (colour = slope severity)
2. Draws inlier dots (bright green, 5px) and outlier dots (red, 4px)
3. If WALL/OBSTACLE: flashes a thick red border around the whole image (alternates every ~4 frames using frame counter)
4. **Compact top-left HUD card** (~240px wide):
   - Purple title bar "ZED2i PLANE VISUALIZER"
   - Rows: Normal vector, Pitch, Camber, Planarity | Inlier count
5. **Bottom status bar** (full-width, 64px tall):
   - Top border coloured by slope severity
   - Large angle number on left (e.g. `-17.4`)
   - Classification label in centre
   - Severity progress bar (0–90°) on right

---

## Current State / Limitations

- The plane is fitted to whatever is in the 3m × 1.5m FOV corridor ahead. It does **not** distinguish ground from floating objects unless RANSAC filters them as outliers.
- The `planarity` score drops when the scene is mixed (e.g. rover is at an intersection of two surfaces). When planarity < 0.50 or inlier_ratio < 0.25 the plane is hidden and "UNCERTAIN" is shown.
- The slope sign (positive = up, negative = down/pit) has been validated against ZED frame conventions but may need re-tuning if the camera mounting angle changes significantly.
- The window opens at 1280×720 with `cv2.WINDOW_NORMAL`. Quit with Q or Esc.

---

## Files in `~/zed_slope/`

| File | Purpose |
|---|---|
| `plane_visualizer.py` | Main ROS2 node (707 lines) — all logic |
| `run.sh` | Shell wrapper: sources ROS2 and runs the Python node |
| `project.md` | Earlier brief technical overview |
| `README.md` | Short usage notes |
