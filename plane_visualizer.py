import sys
import threading
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from collections import deque

try:
    import pyzed.sl as sl
    HAVE_ZED_SDK = True
except ImportError:
    sl = None
    HAVE_ZED_SDK = False

# ── ZED SDK direct-grab mode (run with --sdk; stop the ROS zed wrapper first) ──
USE_SDK           = "--sdk" in sys.argv
SDK_AUTO_EXPOSURE = False   # False = manual exposure/gain (what you want at night)
SDK_EXPOSURE      = 80      # 0-100 %
SDK_GAIN          = 60      # 0-100
SDK_STEP          = 5       # change per key press
SDK_CLOUD_W, SDK_CLOUD_H = 640, 360   # point-cloud resolution (speed vs density)

# ── Remote viewer mode ─────────────────────────────────────────────────────
# Run with --headless on the rover: no local OpenCV window is opened.
# The rendered overlay is published to PUB_TOPIC at PUB_RESIZE resolution.
# On the base laptop: ros2 run rqt_image_view rqt_image_view /slope_visualizer/overlay
#                 OR: python3 viewer.py
HEADLESS      = "--headless" in sys.argv
PUB_TOPIC     = "/slope_visualizer/overlay"  # topic the rover publishes to
PUB_RESIZE    = (960, 540)                   # (w, h) published; reduce for bandwidth

# ── Dense plane fitting ────────────────────────────────────────────────
GRID_M            = 10       # depth cells for grid sampling (only used by the legacy path)
GRID_N            = 10       # lateral cells for grid sampling (only used by the legacy path)
RANSAC_ITERATIONS = 60       # RANSAC iterations
RANSAC_THRESH     = 0.05     # inlier distance threshold in metres (ZED noise grows with range)
MIN_INLIERS       = 15       # minimum inliers to trust RANSAC result
DEPTH_BIN_WIDTH   = 0.15     # histogram bin width for depth isolation (legacy path)
NORMAL_BUF_SIZE   = 10       # smoothing buffer length

# ── Slope thresholds (degrees) ─────────────────────────────────────────
THR_FLAT          = 10.0
THR_GENTLE        = 25.0
THR_STEEP         = 45.0
THR_EXTREME       = 75.0

# ── Confidence gates ───────────────────────────────────────────────────
MIN_PLANARITY     = 0.50
MIN_INLIER_RATIO  = 0.25

# ─────────────────────────── tunable FOV (slope stage) ────────────────────
FOV_DEPTH_MIN  = 3.5    # metres ahead – slope is detected starting from here
FOV_DEPTH_MAX  = 8.0    # metres ahead  (X axis in zed_left_camera_frame)
FOV_LAT_HALF   = 0.75   # ± metres laterally (Y axis)

# ── GROUND-ONLY selection (fix for "plane fitted above the ground") ───────
# MEASURE THIS: height of the ZED left camera above the floor, in metres.
CAM_HEIGHT        = 0.45    # metres: measured ZED2i optical-center height above ground
# Height gate follows the learned near-ground line in camera coordinates, then
# opens from FOV_DEPTH_MIN to admit the expected ramp/descent envelope.
GROUND_TOL_ABOVE  = 0.15    # m   slack above the expected floor
GROUND_TOL_BELOW  = 0.30    # m   slack below the expected floor
GATE_RAMP_DEG     = 30.0    # deg ramp slope that sets how fast the upper gate opens
GATE_DESC_DEG     = 30.0    # deg descent slope that sets how fast the lower gate opens
GATE_MAX_RISE     = 1.50    # m   allow a 30 deg ramp across the far detection span
GATE_MAX_DROP     = 1.50    # m   allow a 30 deg descent across the far detection span
# "Lowest surface per cell": ground is the lowest thing in every (depth, lateral)
# cell; chairs, tables and walls sit ABOVE it, so they are discarded.
GROUND_CELL_D     = 0.25    # m   cell size along depth
GROUND_CELL_L     = 0.15    # m   cell size laterally
GROUND_MIN_CELL_PTS = 5     # points a cell needs to be used
GROUND_PERCENTILE = 10      # percentile of height taken as the cell's floor
GROUND_BAND       = 0.08    # m   keep points within this of the cell floor
GROUND_MIN_CELLS  = 8       # fewer ground cells than this = no plane this frame
# The camera must sit clearly ABOVE the fitted plane. A plane through chair
# seats / walls ends up near or above the camera and is rejected.
GROUND_MIN_CAM_FRAC = 0.6   # plane must be >= this * CAM_HEIGHT below the camera
# A plane fit is accepted only when its RANSAC inliers also track the
# independently built near-to-far ground profile at the same depths.
GROUND_PROFILE_MATCH_TOL = 0.12  # m vertical difference from profile ground height
GROUND_PROFILE_MATCH_BINS = 5    # minimum distinct depth bins supporting the match
# Legacy "dominant depth" step: keeps only the densest 0.3 m depth slice. On a
# cluttered scene that slice is a wall/furniture and the points collapse into a
# line, so tilt along depth is unconstrained. Leave False.
USE_DOMINANT_DEPTH = False
# Draw the RANSAC outliers (small dim red dots)? Off keeps the view clean.
SHOW_OUTLIERS      = False
# ──────────────────────────────────────────────────────────────────────────

# ── Hazard scan (pits / trenches / drop-offs / obstacles ahead) ───────────
HAZ_DEPTH_MIN     = 0.5     # m   where the hazard scan starts
HAZ_DEPTH_MAX     = 5.0     # m   how far ahead it looks
HAZ_STRIP         = 0.10    # m   depth resolution of the ground profile
HAZ_LANES         = 3       # lateral lanes across the corridor
HAZ_MIN_PTS       = 15      # points a lane-strip needs to count as "has data"
HAZ_MIN_GROUND    = 3       # data strips a lane needs before it can be judged
HAZ_GAP_STRIPS    = 3       # empty strips in a row (0.3 m) = missing ground
HAZ_END_MARGIN    = 0.6     # m   ignore "ground ends" this close to HAZ_DEPTH_MAX
HAZ_STEP_DROP     = 0.12    # m   sudden drop between neighbouring strips
HAZ_RISE_LIMIT    = 0.30    # m   ground that already rose this much = ramp, not pit
HAZ_BELOW_MAX     = 0.60    # m   ground-profile band: ignore points deeper than this
HAZ_ABOVE_MAX     = 1.00    # m   ignore points higher than this (ceiling, trees)
HAZ_PIT_DEPTH     = 0.15    # m   surface this far below LOCAL ground = evidence of a pit
HAZ_FLOOR_MAX     = 2.00    # m   ignore returns deeper than this under the plane
HAZ_FLOOR_PTS     = 40      # pts that must lie below ground inside a gap to confirm a pit
HAZ_PLANE_TOL     = 0.06    # m   a strip is 'on the ground plane' if |height| < this ...
HAZ_PLANE_TOL_M   = 0.03    # m   ... plus this per metre of range (plane tilt error grows with range)
HAZ_FLAT_RUN      = 5       # strips (0.5 m) of flat surface = a platform the rover can stand on
HAZ_FLAT_TOL      = 0.04    # m   max height wobble along a flat run
HAZ_FLAT_SPREAD   = 0.10    # m   max height spread inside one strip of a flat run
HAZ_STEP_DROP_FLAT = 0.15   # m   drop needed when the 'standing surface' is a flat run, not the plane
OBST_H_MIN        = 0.20    # m   structure must rise this far above the ground
OBST_SPREAD       = 0.25    # m   ... and span this much height inside one strip
HAZ_REF_MAX_TILT  = 35.0    # deg only scan if the reference plane looks like ground
HAZ_HIST          = 5       # frames remembered
HAZ_CONFIRM       = 3       # frames (of HAZ_HIST) that must agree
HAZ_HOLD_FRAMES   = 20      # keep warning this long if the reference plane is lost
HAZ_FLASH_DIST    = 3.0     # m   flash the border when closer than this
HAZ_BAND_LEN      = 0.5     # m   length of the red band drawn on the ground
HAZ_MAX_PTS       = 80000   # subsample cap for speed
HAZ_MIN_CLOUD     = 300     # fewer corridor points than this = skip scan
# ──────────────────────────────────────────────────────────────────────────

# BGR colours (OpenCV order)
COL_PLANE  = (0,  180, 255)   # amber
COL_P = [
    (60,  60,  255),           # P1 red
    (60,  220,  60),           # P2 green
    (255, 100,  60),           # P3 blue
]
COL_NORMAL = (0, 240, 240)    # cyan
COL_HUD_BG = (15,  15,  15)
COL_HUD_FG = (220, 220, 220)
COL_HAZ_PIT  = (0, 0, 255)      # red
COL_HAZ_OBST = (0, 140, 255)    # orange

PLANE_HALF_SIZE = 0.9   # half-size of the FALLBACK plane square in metres (confident path uses extended rect)

# ── Extended plane drawing ─────────────────────────────────────────────────
# The fitted plane is extended BACKWARD to PLANE_DRAW_NEAR so there is no
# blind-spot between the rover and the detection range (FOV_DEPTH_MIN).
PLANE_DRAW_NEAR  = 0.5    # m  how close the drawn plane comes to the rover
# PLANE_DRAW_FAR is taken from FOV_DEPTH_MAX at render time.

# ── Along-plane slope profile ───────────────────────────────────────────────
# Stats are calculated from the current frame, across depth, over the drawn
# ground range. Each local slope uses a ~1.5 m window.
PROFILE_CELL_D       = GROUND_CELL_D
PROFILE_MIN_CELLS    = 7       # depth bins per local fit (~1.5 m baseline)
PROFILE_MIN_PTS_CELL = 2
PROFILE_MIN_WINDOW_PTS = PROFILE_MIN_CELLS - 1  # allow one missing bin
PROFILE_MIN_SPAN_M   = 1.25    # minimum depth span when one bin is missing
PROFILE_MAX_STEP     = 0.30    # max height change per 0.25 m depth cell
PROFILE_MAX_GAP      = 2       # stop after this many consecutive empty bins
PROFILE_START_TOL    = 0.10    # nearest accepted bin must be near configured floor height
PROFILE_ABS_ABOVE    = 2.0     # loose ceiling/junk gate above nominal floor
PROFILE_ABS_BELOW    = 2.0     # loose far/junk gate below nominal floor
PROFILE_HEIGHT_SMOOTH_CELLS = 3  # median filter depth-bin heights before fitting
PROFILE_BASELINE_FAR = 2.0       # internal zero-grade reference; stats still span full plane
PROFILE_BASELINE_MIN_PTS = 4     # samples required to estimate the near-ground grade
PROFILE_BASELINE_RMS_MAX = 0.02  # near reference surface must be straight
PROFILE_BASELINE_HALF_DIFF_MAX = 2.0  # near/far half slopes must agree (degrees)
PROFILE_BASELINE_INIT_MAX_ABS = 12.0  # reject a ramp as the startup zero reference
PROFILE_SEG_MEDIAN = 3          # median neighbouring local slopes by depth
PROFILE_MAX_SUPPORT_DELTA = 4.0 # max slope needs a neighbouring window within this many degrees
PROFILE_STATS_HISTORY = 7        # temporal median history (single temporal filter)
PROFILE_STATS_HOLD_FRAMES = 10   # keep last stats this many missing cloud frames

_PROFILE_STATE = {"baseline_deg": None, "baseline_intercept": None}

SKIP = object()         # "hazard scan could not run this frame"
LAST_PROFILE = None     # per-lane profile of the last hazard scan (press D to print)
LAST_SLOPE_PROFILE = None  # local ground-slope segments (press D to print)


# ──────────────────────────── helpers ─────────────────────────────────────

def select_ground_points(pts: np.ndarray, depth_col: int, lat_col: int,
                         up_col: int, up_sign: float) -> np.ndarray:
    """
    Return one representative point per (depth, lateral) cell, taken from the
    LOWEST surface in that cell.  Ground is the lowest thing in every cell;
    furniture, people and walls sit above it, so they are rejected.  This is
    what keeps the plane fit on the floor instead of on chair seats / walls.

    The grid is built from the actual range of the input points, so the same
    function works for both the far detection zone and the near display zone.
    """
    if pts.shape[0] == 0:
        return np.empty((0, 3))

    dep = pts[:, depth_col]
    lat = pts[:, lat_col]
    ht  = pts[:, up_col] * up_sign          # height relative to camera (up = +)

    # Anchor grid cells to fixed world-frame multiples so small point-extent
    # changes cannot shift the sampling boundaries from frame to frame.
    dep_min, dep_max = float(dep.min()), float(dep.max())
    lat_min, lat_max = float(lat.min()), float(lat.max())
    dep_min = np.floor(dep_min / GROUND_CELL_D) * GROUND_CELL_D
    lat_min = np.floor(lat_min / GROUND_CELL_L) * GROUND_CELL_L

    nd = max(1, int(np.floor((dep_max - dep_min) / GROUND_CELL_D)) + 1)
    nl = max(1, int(np.floor((lat_max - lat_min) / GROUND_CELL_L)) + 1)
    di = np.clip(((dep - dep_min) / GROUND_CELL_D).astype(int), 0, nd - 1)
    li = np.clip(((lat - lat_min) / GROUND_CELL_L).astype(int), 0, nl - 1)
    cell = di * nl + li

    order = np.argsort(cell, kind="stable")
    cs = cell[order]
    _, first = np.unique(cs, return_index=True)
    last = np.append(first[1:], cs.size)

    out = []
    for a, b in zip(first, last):
        if b - a < GROUND_MIN_CELL_PTS:
            continue
        idx = order[a:b]
        h = ht[idx]
        floor_h = np.percentile(h, GROUND_PERCENTILE)
        low = idx[h <= floor_h + GROUND_BAND]
        if low.size == 0:
            continue
        hl = ht[low]
        out.append(pts[low[np.argmin(np.abs(hl - np.median(hl)))]])

    return np.array(out) if out else np.empty((0, 3))


def plane_profile_slopes(ground_pts: np.ndarray, depth_col: int, up_col: int,
                         up_sign: float):
    """Build a continuous near-to-far profile and fit local grade windows."""
    stored_baseline = _PROFILE_STATE["baseline_deg"]
    baseline_report = float(stored_baseline) if stored_baseline is not None else 0.0
    stored_intercept = _PROFILE_STATE["baseline_intercept"]
    intercept_report = float(stored_intercept) if stored_intercept is not None else -CAM_HEIGHT

    def empty_result(reason):
        return {"stats": None, "segments": [], "accepted_bins": {},
                "walk_stop": reason, "baseline_deg": baseline_report,
                "baseline_intercept": intercept_report,
                "baseline_valid": stored_baseline is not None,
                "bridged_gaps": 0}

    if ground_pts.shape[0] == 0:
        return empty_result("no valid bins")

    dep = ground_pts[:, depth_col]
    height = ground_pts[:, up_col] * up_sign
    first_bin = int(np.floor(PLANE_DRAW_NEAR / PROFILE_CELL_D))
    last_bin = int(np.ceil(FOV_DEPTH_MAX / PROFILE_CELL_D))
    bins = np.floor(dep / PROFILE_CELL_D).astype(int)

    # Median across lateral cells makes each depth strip one robust height
    # sample. Require multiple returns so isolated points do not set the grade.
    profile = {}
    profile_counts = {}
    for idx in range(first_bin, last_bin):
        h = height[bins == idx]
        if h.size >= PROFILE_MIN_PTS_CELL:
            profile[idx] = float(np.median(h))
            profile_counts[idx] = int(h.size)

    # Start at the nearest valid depth and walk outward. Short gaps are
    # tolerated; a height discontinuity ends the ground track (e.g. a wall).
    valid = sorted(profile)
    if not valid:
        return empty_result("no valid bins")

    nominal_height = -CAM_HEIGHT
    start_idx = next((idx for idx in valid
                      if abs(profile[idx] - nominal_height) <= PROFILE_START_TOL), None)
    if start_idx is None:
        return empty_result(f"no start bin within {PROFILE_START_TOL:.2f} m of nominal floor")

    accepted = {start_idx: profile[start_idx]}
    last_idx = start_idx
    empty_run = 0
    bridged_gaps = 0
    walk_stop = f"reached far edge ({FOV_DEPTH_MAX:.2f} m)"
    for idx in range(last_idx + 1, last_bin):
        if idx not in profile:
            empty_run += 1
            if empty_run > PROFILE_MAX_GAP:
                walk_stop = f"gap at {idx * PROFILE_CELL_D:.2f} m"
                break
            continue
        gap_cells = idx - last_idx
        if abs(profile[idx] - profile[last_idx]) > PROFILE_MAX_STEP * gap_cells:
            empty_run += 1
            if empty_run > PROFILE_MAX_GAP:
                walk_stop = f"step violation at {idx * PROFILE_CELL_D:.2f} m"
                break
            continue
        if empty_run:
            bridged_gaps += empty_run
        accepted[idx] = profile[idx]
        last_idx = idx
        empty_run = 0

    # Median-filter height along depth before fitting. This suppresses isolated
    # depth speckle without clipping real large slope values.
    radius = PROFILE_HEIGHT_SMOOTH_CELLS // 2
    smooth = {}
    for idx in accepted:
        vals = [accepted[k] for k in range(idx - radius, idx + radius + 1)
                if k in accepted]
        smooth[idx] = float(np.median(vals))

    # Estimate the camera/frame pitch only from a straight, locally consistent
    # near-ground reference. A ramp foot entering this zone must not rotate the
    # zero reference and subtract its grade from every profile segment.
    baseline = [(idx, h) for idx, h in smooth.items()
                if (idx + 0.5) * PROFILE_CELL_D <= PROFILE_BASELINE_FAR]
    candidate_deg = None
    candidate_intercept = None
    baseline_good = False
    if len(baseline) >= PROFILE_BASELINE_MIN_PTS:
        bx = np.asarray([(idx + 0.5) * PROFILE_CELL_D for idx, _ in baseline])
        by = np.asarray([h for _, h in baseline])
        if np.all(np.isfinite(bx)) and np.all(np.isfinite(by)):
            full_fit = np.polyfit(bx, by, 1)
            residual = by - np.polyval(full_fit, bx)
            rms = float(np.sqrt(np.mean(residual ** 2)))
            near = bx < 1.25
            far = bx >= 1.25
            if np.count_nonzero(near) >= 2 and np.count_nonzero(far) >= 2:
                near_fit = np.polyfit(bx[near], by[near], 1)
                far_fit = np.polyfit(bx[far], by[far], 1)
                near_deg = float(np.degrees(np.arctan(near_fit[0])))
                far_deg = float(np.degrees(np.arctan(far_fit[0])))
                candidate_deg = float(np.degrees(np.arctan(full_fit[0])))
                candidate_intercept = float(full_fit[1])
                baseline_good = (
                    np.isfinite(rms) and rms <= PROFILE_BASELINE_RMS_MAX and
                    abs(near_deg - far_deg) <= PROFILE_BASELINE_HALF_DIFF_MAX
                )

    # Learn the camera-frame floor reference once. Never relearn it from a
    # later ramp that happens to fill the reference window and look straight.
    if (_PROFILE_STATE["baseline_deg"] is None and baseline_good and
            candidate_deg is not None and candidate_intercept is not None and
            abs(candidate_deg) <= PROFILE_BASELINE_INIT_MAX_ABS):
        _PROFILE_STATE["baseline_deg"] = candidate_deg
        _PROFILE_STATE["baseline_intercept"] = candidate_intercept

    baseline_deg = _PROFILE_STATE["baseline_deg"]
    baseline_intercept = _PROFILE_STATE["baseline_intercept"]
    baseline_available = baseline_deg is not None
    baseline_report = float(baseline_deg) if baseline_available else 0.0
    intercept_report = float(baseline_intercept) if baseline_intercept is not None else -CAM_HEIGHT
    if not baseline_available:
        walk_stop = ("insufficient/curved near-ground baseline "
                     f"(need {PROFILE_BASELINE_MIN_PTS} bins; RMS <= "
                     f"{PROFILE_BASELINE_RMS_MAX:.2f} m and half slopes within "
                     f"{PROFILE_BASELINE_HALF_DIFF_MAX:.1f} deg; startup zero must "
                     f"be within {PROFILE_BASELINE_INIT_MAX_ABS:.1f} deg)")

    # Fit overlapping ~1.5 m windows. A window may omit one depth bin if its
    # remaining samples still span at least 1.25 m.
    segments = []
    for start in range(first_bin, last_bin - PROFILE_MIN_CELLS + 1):
        items = [(idx, accepted[idx]) for idx in range(start, start + PROFILE_MIN_CELLS)
                 if idx in accepted]
        if len(items) < PROFILE_MIN_WINDOW_PTS:
            continue
        x = np.asarray([(idx + 0.5) * PROFILE_CELL_D for idx, _ in items])
        y = np.asarray([smooth[idx] for idx, _ in items])
        span = float(x[-1] - x[0])
        if span < PROFILE_MIN_SPAN_M:
            continue
        counts = np.asarray([profile_counts[idx] for idx, _ in items], dtype=float)
        weights = np.sqrt(np.maximum(counts, 1.0)) / np.maximum(x, 1.0) ** 2
        if (not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or
                not np.all(np.isfinite(weights)) or np.max(weights) <= 0):
            continue
        weights /= np.max(weights)
        grade = float(np.polyfit(x, y, 1, w=weights)[0])
        if not np.isfinite(grade):
            continue
        raw_slope = float(np.degrees(np.arctan(grade)))
        segments.append({
            "depth_start": float(x[0] - PROFILE_CELL_D / 2),
            "depth_end": float(x[-1] + PROFILE_CELL_D / 2),
            "point_count": int(sum(profile_counts[idx] for idx, _ in items)),
            "raw_slope": raw_slope,
            "slope": float(raw_slope - baseline_deg)
                     if baseline_available else None,
            "bins": len(items),
        })

    segments.sort(key=lambda seg: seg["depth_start"])
    if baseline_available and segments:
        raw_values = np.asarray([seg["slope"] for seg in segments], dtype=float)
        camera_values = np.asarray([seg["raw_slope"] for seg in segments], dtype=float)
        half = PROFILE_SEG_MEDIAN // 2
        values = np.asarray([
            np.median(raw_values[max(0, i - half):min(len(raw_values), i + half + 1)])
            for i in range(len(raw_values))
        ], dtype=float)
        camera_values = np.asarray([
            np.median(camera_values[max(0, i - half):min(len(camera_values), i + half + 1)])
            for i in range(len(camera_values))
        ], dtype=float)
        for seg, value, raw_value in zip(segments, values, camera_values):
            seg["slope"] = float(value)
            seg["raw_slope"] = float(raw_value)
        # MAX is especially sensitive to one noisy depth window. Keep only
        # peaks corroborated by an adjacent overlapping window; a real ramp
        # spanning multiple windows remains eligible, while isolated spikes
        # do not set the displayed maximum.
        supported = [
            i for i, value in enumerate(values)
            if ((i > 0 and abs(float(value - values[i - 1])) <= PROFILE_MAX_SUPPORT_DELTA) or
                (i + 1 < len(values) and abs(float(value - values[i + 1])) <= PROFILE_MAX_SUPPORT_DELTA))
        ]
        if supported:
            max_idx = max(supported, key=lambda i: values[i])
        else:
            # With only one unsupported peak, report the median profile grade
            # instead of presenting that single window as a confirmed MAX.
            max_idx = int(np.argsort(values)[len(values) // 2])
        raw_supported_max = float(camera_values[max_idx])
        stats = {
            "min": float(values.min()),
            "avg": float(values.mean()),
            "max": float(values[max_idx]),
            "raw_min": float(camera_values.min()),
            "raw_avg": float(camera_values.mean()),
            "raw_max": raw_supported_max,
        }
    else:
        stats = None
    return {
        "stats": stats,
        "segments": segments,
        "accepted_bins": accepted,
        "walk_stop": walk_stop,
        "baseline_deg": baseline_report,
        "baseline_intercept": intercept_report,
        "baseline_valid": baseline_available,
        "bridged_gaps": bridged_gaps,
    }


def grid_sample_fov(pts: np.ndarray, depth_col: int, lat_col: int, grid_m=GRID_M, grid_n=GRID_N) -> np.ndarray:
    """Legacy sampler (median-depth point per cell). Kept for reference / fallback."""
    if pts.shape[0] == 0:
        return np.empty((0, 3))
    d_min, d_max = pts[:, depth_col].min(), pts[:, depth_col].max()
    l_min, l_max = pts[:, lat_col].min(), pts[:, lat_col].max()
    d_step = (d_max - d_min) / grid_m
    l_step = (l_max - l_min) / grid_n

    sampled = []
    # add small epsilon to avoid out of bounds on exactly max value
    d_idx = np.clip(((pts[:, depth_col] - d_min) / (d_step + 1e-9)).astype(int), 0, grid_m - 1)
    l_idx = np.clip(((pts[:, lat_col] - l_min) / (l_step + 1e-9)).astype(int), 0, grid_n - 1)

    for i in range(grid_m):
        for j in range(grid_n):
            mask = (d_idx == i) & (l_idx == j)
            if np.any(mask):
                cell_pts = pts[mask]
                # median point in cell based on depth
                d_med = np.median(cell_pts[:, depth_col])
                idx = np.argmin(np.abs(cell_pts[:, depth_col] - d_med))
                sampled.append(cell_pts[idx])
    return np.array(sampled)

def fit_plane_svd(pts: np.ndarray, depth_col: int):
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    n = Vt[-1]

    # Orient normal to point UPWARD (not forward), so the sign of the
    # depth component carries slope direction information.
    # In zed_left: up = +Z (col 2). In optical: up = -Y (col 1, sign -1).
    if depth_col == 2:   # optical
        up_col, up_sign = 1, -1.0
    else:                # zed_left
        up_col, up_sign = 2,  1.0

    if n[up_col] * up_sign < 0:   # normal pointing down → flip
        n = -n

    planarity_score = 1.0 - (S[2] / S[0]) if S[0] > 1e-9 else 0.0
    d = float(np.dot(n, centroid))
    return n, d, centroid, planarity_score

def ransac_plane(pts: np.ndarray, depth_col: int, n_iter=RANSAC_ITERATIONS, inlier_thresh=RANSAC_THRESH):
    N = pts.shape[0]
    if N < 3:
        return None

    best_inlier_count = 0
    best_inlier_mask = None

    for _ in range(n_iter):
        idxs = np.random.choice(N, 3, replace=False)
        p1, p2, p3 = pts[idxs]
        v1, v2 = p2 - p1, p3 - p1
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm

        d = np.dot(n, p1)
        # distance of all points to plane
        dists = np.abs(np.dot(pts, n) - d)
        inlier_mask = dists < inlier_thresh
        inlier_count = np.sum(inlier_mask)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inlier_mask = inlier_mask

    if best_inlier_mask is None or best_inlier_count < MIN_INLIERS:
        return None

    # Refit SVD on inliers
    inliers = pts[best_inlier_mask]
    n, d, centroid, planarity = fit_plane_svd(inliers, depth_col)

    # re-evaluate mask with refined plane (optional, but good practice)
    dists = np.abs(np.dot(pts, n) - d)
    refined_mask = dists < inlier_thresh
    return n, d, centroid, planarity, refined_mask

def find_dominant_depth(pts: np.ndarray, depth_col: int, bin_width=DEPTH_BIN_WIDTH):
    if pts.shape[0] == 0:
        return np.ones(0, dtype=bool)

    depths = pts[:, depth_col]
    d_min, d_max = depths.min(), depths.max()
    bins = np.arange(d_min, d_max + bin_width, bin_width)
    if len(bins) < 2:
        return np.ones(pts.shape[0], dtype=bool)

    hist, edges = np.histogram(depths, bins=bins)
    peak_idx = np.argmax(hist)
    peak_center = (edges[peak_idx] + edges[peak_idx+1]) / 2.0

    mask = np.abs(depths - peak_center) <= bin_width
    return mask

def compute_slope(n: np.ndarray, depth_col: int):
    """
    Compute signed slope angle relative to horizontal.

    Sign convention (rover-centric):
      +  positive angle  → surface tilts UPWARD ahead  (hill/ramp)
      -  negative angle  → surface tilts DOWNWARD ahead (pit/trench/descent)
    """
    if depth_col == 2:   # optical frame
        up_col, up_sign = 1, -1.0
    else:                # zed_left_camera_frame
        up_col, up_sign = 2,  1.0

    n_ref = np.zeros(3)
    n_ref[up_col] = up_sign   # unit "up" vector

    # Absolute angle from horizontal (always 0-90)
    dot_up = np.clip(np.dot(n_ref, n), 0.0, 1.0)   # n already points up
    abs_theta = float(np.degrees(np.arccos(dot_up)))

    # Sign from depth component: forward-leaning normal = depression
    signed_theta = -abs_theta if n[depth_col] > 0 else abs_theta

    # Pitch and camber
    lat_col = 0 if depth_col == 2 else 1
    pitch_deg  = float(np.degrees(np.arctan2(n[depth_col], dot_up + 1e-9)))
    camber_deg = float(np.degrees(np.arctan2(n[lat_col],   dot_up + 1e-9)))

    a = abs(signed_theta)
    if a <= THR_FLAT:
        cls, col = ("FLAT (DIP)", (0, 200, 150)) if signed_theta < -1.0 else ("FLAT", (0, 200, 0))
    elif a <= THR_GENTLE:
        cls, col = ("GENTLE DESCENT", (0, 215, 255)) if signed_theta < 0 else ("GENTLE SLOPE", (0, 200, 255))
    elif a <= THR_STEEP:
        cls, col = ("STEEP DESCENT",  (0, 100, 255)) if signed_theta < 0 else ("STEEP SLOPE",  (0, 140, 255))
    elif a <= THR_EXTREME:
        cls, col = ("PIT / TRENCH",   (0, 0, 220))   if signed_theta < 0 else ("EXTREME SLOPE",(0, 0, 255))
    else:
        cls, col = "WALL / OBSTACLE", (0, 0, 255)

    return {
        "theta": signed_theta,
        "abs_theta": abs_theta,
        "pitch": pitch_deg,
        "camber": camber_deg,
        "classification": cls,
        "colour": col
    }


# ───────────────────────── hazard scan ────────────────────────────────────

def analyze_hazards(pts_raw: np.ndarray, n: np.ndarray, d: float,
                    depth_col: int, lat_col: int):
  
    dep_all = pts_raw[:, depth_col]
    lat_all = pts_raw[:, lat_col]
    m = ((dep_all > HAZ_DEPTH_MIN) & (dep_all < HAZ_DEPTH_MAX) &
         (np.abs(lat_all) < FOV_LAT_HALF))
    c = pts_raw[m]
    if c.shape[0] < HAZ_MIN_CLOUD:
        return SKIP, None
    if c.shape[0] > HAZ_MAX_PTS:
        c = c[:: c.shape[0] // HAZ_MAX_PTS + 1]

    h = c @ n - d                                   # height above ground plane
    keep = (h > -HAZ_FLOOR_MAX) & (h < HAZ_ABOVE_MAX)
    c, h = c[keep], h[keep]
    if c.shape[0] < HAZ_MIN_CLOUD // 2:
        return SKIP, None

    ns = int(round((HAZ_DEPTH_MAX - HAZ_DEPTH_MIN) / HAZ_STRIP))
    W = FOV_LAT_HALF
    k_idx = np.clip(((c[:, depth_col] - HAZ_DEPTH_MIN) / HAZ_STRIP).astype(int), 0, ns - 1)
    l_idx = np.clip(((c[:, lat_col] + W) / (2 * W) * HAZ_LANES).astype(int), 0, HAZ_LANES - 1)

    # ground-profile statistics use only points inside the normal band
    g = h > -HAZ_BELOW_MAX
    cell = (l_idx * ns + k_idx)[g]
    hg = h[g]

    order = np.argsort(cell, kind="stable")
    cs = cell[order]
    uniq, first = np.unique(cs, return_index=True)
    last = np.append(first[1:], cs.size)

    cnt = np.zeros((HAZ_LANES, ns), dtype=int)
    lo  = np.full((HAZ_LANES, ns), np.nan)
    med = np.full((HAZ_LANES, ns), np.nan)
    hi  = np.full((HAZ_LANES, ns), np.nan)
    for u, a, b in zip(uniq, first, last):
        ln, k = divmod(int(u), ns)
        cnt[ln, k] = b - a
        if b - a >= HAZ_MIN_PTS:
            lo[ln, k], med[ln, k], hi[ln, k] = np.percentile(hg[order[a:b]], (10, 50, 90))

    def depth_of(k):
        return HAZ_DEPTH_MIN + k * HAZ_STRIP

    found = []        # (depth, type, lane, why)
    unverified = []   # (depth, lane)

    for ln in range(HAZ_LANES):
        idx = np.flatnonzero(cnt[ln] >= HAZ_MIN_PTS)
        if idx.size < HAZ_MIN_GROUND:
            continue

        g0 = np.median(med[ln, idx[:3]])            # local ground level (nearest strips)
        with np.errstate(invalid="ignore"):
            is_obst = (cnt[ln] >= HAZ_MIN_PTS) & ((hi[ln] - g0) > OBST_H_MIN) & \
                      ((hi[ln] - lo[ln]) > OBST_SPREAD)

        if is_obst.any():
            k0 = int(np.argmax(is_obst))
            found.append((depth_of(k0), "OBSTACLE", ln, f"rises {hi[ln, k0] - g0:.2f} m"))

        # A pit / drop-off needs a trustworthy "standing surface" just before the edge:
        #   (a) it lies on the slope-stage ground plane, or
        #   (b) it is a flat platform (>= HAZ_FLAT_RUN strips, small wobble and spread).
        # (b) covers a stair going down when the fitted plane is pulled off the surface
        # the rover stands on.  An object (chair, pet) is neither: it sits off the plane
        # and is rough, so object-then-floor is not a drop.
        def on_plane(k):
            return abs(med[ln, k]) <= HAZ_PLANE_TOL + HAZ_PLANE_TOL_M * depth_of(k)

        def flat_run(k):
            n_ok, kk = 0, k
            while kk >= 0 and cnt[ln, kk] >= HAZ_MIN_PTS:
                if abs(med[ln, kk] - med[ln, k]) > HAZ_FLAT_TOL or (hi[ln, kk] - lo[ln, kk]) > HAZ_FLAT_SPREAD:
                    break
                n_ok += 1
                kk -= 1
            return n_ok >= HAZ_FLAT_RUN

        def standing(k):
            return on_plane(k) or flat_run(k)

        def gap_is_pit(prev):
            if not standing(prev):                           # not standing on real ground
                return False
            if med[ln, prev] - g0 > HAZ_RISE_LIMIT:          # ground was climbing: ramp
                return False
            if is_obst[max(0, prev - 2): prev + 1].any():    # hidden behind an obstacle
                return False
            return True

        def floor_evidence(prev, end):
            """Points clearly below the local ground inside gap strips prev+1..end."""
            sel = (l_idx == ln) & (k_idx > prev) & (k_idx <= end)
            hh = h[sel]
            low = hh[hh < med[ln, prev] - HAZ_PIT_DEPTH]
            if low.size >= HAZ_FLOOR_PTS:
                return f"floor {med[ln, prev] - np.median(low):.2f} m lower, {low.size} pts"
            return None

        prev = int(idx[0])
        done = False
        for p in range(1, len(idx)):
            j = int(idx[p])
            gap = j - prev - 1
            if gap >= HAZ_GAP_STRIPS:
                if gap_is_pit(prev):
                    step = med[ln, j] - med[ln, prev]
                    need = HAZ_PIT_DEPTH if on_plane(prev) else HAZ_STEP_DROP_FLAT
                    ptype = "PIT / TRENCH"
                    if step < -need:
                        why = f"ground resumes {-step:.2f} m lower"
                        ptype = "DROP-OFF"                   # lip hides the step, ground beyond is lower
                    else:
                        why = floor_evidence(prev, j - 1)
                    if why:
                        found.append((depth_of(prev + 1), ptype, ln, why))
                        done = True
                        break
                    unverified.append((depth_of(prev + 1), ln))
            else:
                step = med[ln, j] - med[ln, prev]
                # only a step DOWN FROM the ground plane counts; an object (chair, pet,
                # box) followed by floor also looks like a "step down" and must not.
                need = HAZ_STEP_DROP if on_plane(prev) else HAZ_STEP_DROP_FLAT
                if step < -need and standing(prev) and not is_obst[max(0, prev - 2): prev + 1].any():
                    nxt = int(idx[p + 1]) if p + 1 < len(idx) else None
                    # the step must persist into the next strip (not one noisy strip)
                    if nxt is None or med[ln, nxt] - med[ln, prev] < -HAZ_STEP_DROP:
                        found.append((depth_of(j), "DROP-OFF", ln, f"step down {-step:.2f} m"))
                        done = True
                        break
            prev = j

        if not done:
            trailing = ns - 1 - prev
            if (trailing >= HAZ_GAP_STRIPS and
                    depth_of(prev + 1) <= HAZ_DEPTH_MAX - HAZ_END_MARGIN and
                    gap_is_pit(prev)):
                why = floor_evidence(prev, ns - 1)
                if why:
                    found.append((depth_of(prev + 1), "PIT / TRENCH", ln, why))
                else:
                    unverified.append((depth_of(prev + 1), ln))

    global LAST_PROFILE
    LAST_PROFILE = {"cnt": cnt, "med": med, "lo": lo, "hi": hi, "found": list(found)}

    note = None
    if unverified:
        note = f"no-data gap @{min(unverified)[0]:.1f}m, unverified"

    if not found:
        return None, note

    best = min(found, key=lambda f: f[0])              # nearest hazard
    lanes = [f[2] for f in found if abs(f[0] - best[0]) <= 0.5]
    lane_w = 2 * W / HAZ_LANES
    return {
        "type": best[1],
        "depth": float(best[0]),
        "lat_lo": float(-W + min(lanes) * lane_w),
        "lat_hi": float(-W + (max(lanes) + 1) * lane_w),
        "why": best[3],
    }, note


def print_lane_profile():
    """Debug (key D): print slope segments and the last hazard lane scan."""
    P_slope = LAST_SLOPE_PROFILE
    if P_slope is not None:
        print(f"\n[slope profile] near-ground zero: {P_slope.get('baseline_deg', 0.0):+.2f} deg")
        print(f"[slope profile] walk stopped: {P_slope.get('walk_stop', 'unknown')} | "
              f"bridged gaps: {P_slope.get('bridged_gaps', 0)}")
        if P_slope["accepted_bins"]:
            print("  accepted bins: depth m | height m")
            for idx, height in sorted(P_slope["accepted_bins"].items()):
                print(f"    {(idx + 0.5) * PROFILE_CELL_D:.2f} | {height:+.3f}")
        else:
            print("  accepted bins: none")
        print("  segments: depth range m | points | corrected / raw slope | bins")
        for seg in P_slope["segments"]:
            slope_text = (f"{seg['slope']:+6.2f} deg" if seg["slope"] is not None
                          else "    n/a")
            raw_text = f"{seg.get('raw_slope', float('nan')):+6.2f} deg"
            print(f"  {seg['depth_start']:.2f}–{seg['depth_end']:.2f} | "
                  f"{seg['point_count']:6d} | {slope_text} / {raw_text} | {seg['bins']}")
    else:
        print("\n[slope profile] no profile result in the latest cloud")

    P = LAST_PROFILE
    if P is None:
        print("[hazard profile] no hazard scan has run yet")
        return
    names = [f"lane {i}" for i in range(HAZ_LANES)]
    print("\n[hazard profile] depth m | pts | median h | p10 | p90      (h = height above plane)")
    for ln in range(HAZ_LANES):
        print(f"  -- {names[ln]} --")
        for k in range(P["cnt"].shape[1]):
            if P["cnt"][ln, k] > 0:
                tag = "" if P["cnt"][ln, k] >= HAZ_MIN_PTS else "  (too few pts)"
                print(f"   {HAZ_DEPTH_MIN + k*HAZ_STRIP:4.1f} | {P['cnt'][ln,k]:5d} | "
                      f"{P['med'][ln,k]:+.2f} | {P['lo'][ln,k]:+.2f} | {P['hi'][ln,k]:+.2f}{tag}")
    print("  found:", P["found"])


def hazard_band_3d(hz: dict, n: np.ndarray, d: float, depth_col: int, lat_col: int) -> np.ndarray:
    """4 corners of a band lying on the ground plane at the hazard (native frame)."""
    up_col = 1 if depth_col == 2 else 2
    corners = []
    for D, L in ((hz["depth"], hz["lat_lo"]), (hz["depth"], hz["lat_hi"]),
                 (hz["depth"] + HAZ_BAND_LEN, hz["lat_hi"]), (hz["depth"] + HAZ_BAND_LEN, hz["lat_lo"])):
        p = np.zeros(3)
        p[depth_col] = D
        p[lat_col] = L
        p[up_col] = (d - n[depth_col] * D - n[lat_col] * L) / (n[up_col] if abs(n[up_col]) > 1e-6 else 1e-6)
        corners.append(p)
    return np.array(corners)


def three_fixed_points(pts: np.ndarray, lat_col: int = 1) -> np.ndarray:
   
    depth_col = 0 if lat_col != 0 else 2

    targets = [
        (0.5, 0.0),    # P1 (top/closest)
        (1.0, -0.3),   # P2 (bottom left)
        (1.0, 0.3)     # P3 (bottom right)
    ]

    selected = []
    for (t_depth, t_lat) in targets:
        dist_sq = (pts[:, depth_col] - t_depth)**2 + (pts[:, lat_col] - t_lat)**2
        # Average points within 5cm radius for stability against noise
        min_dist = np.min(dist_sq)
        if min_dist < 0.1**2:
            mask = dist_sq < (np.sqrt(min_dist) + 0.05)**2
            pt = pts[mask].mean(axis=0)
            selected.append(pt)
        else:
            idx = np.argmin(dist_sq)
            selected.append(pts[idx])

    return np.array(selected)


def plane_from_3pts(p1, p2, p3):
    """Return (unit_normal, d) for plane n·x = d.  Normal faces camera."""
    v1, v2 = p2 - p1, p3 - p1
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return np.array([0., 0., 1.]), 0.0
    n /= norm
    if n[0] < 0:      # ensure normal points toward camera (+X direction)
        n = -n
    return n, float(np.dot(n, p1))


def camera_to_optical(pts: np.ndarray) -> np.ndarray:
    """
    Transform points from zed_left_camera_frame  to optical frame.
    camera_frame : X forward, Y left,  Z up
    optical_frame: X right,  Y down,  Z forward
      optical_x = -camera_y
      optical_y = -camera_z
      optical_z =  camera_x
    """
    out = np.empty_like(pts)
    out[:, 0] = -pts[:, 1]   # right  = -left
    out[:, 1] = -pts[:, 2]   # down   = -up
    out[:, 2] =  pts[:, 0]   # fwd    = fwd
    return out


def project_pts(pts_cam: np.ndarray, K: np.ndarray) -> np.ndarray | None:
    """
    Project (N,3) 3-D points in zed_left_camera_frame to (N,2) pixel coords.
    Returns None if any point is behind camera.
    """
    opt = camera_to_optical(pts_cam)
    if np.any(opt[:, 2] <= 0):
        return None
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = (fx * opt[:, 0] / opt[:, 2] + cx).astype(int)
    v = (fy * opt[:, 1] / opt[:, 2] + cy).astype(int)
    return np.stack([u, v], axis=1)


def plane_corners_3d(pts3: np.ndarray, n: np.ndarray, half: float) -> np.ndarray:
    """Build 4 corners of a square plane centred at the centroid of pts3."""
    if len(pts3) == 0:
        return np.empty((0, 3))
    centroid = pts3.mean(axis=0)
    # tangent vectors in the plane
    ref = np.array([0., 0., 1.])
    if abs(np.dot(n, ref)) > 0.9:
        ref = np.array([0., 1., 0.])
    t1 = np.cross(n, ref);  t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1);   t2 /= np.linalg.norm(t2)
    return np.array([
        centroid + half * t1 + half * t2,
        centroid - half * t1 + half * t2,
        centroid - half * t1 - half * t2,
        centroid + half * t1 - half * t2,
    ])


def ground_plane_rect_3d(n: np.ndarray, d: float,
                         depth_col: int, lat_col: int,
                         d_near: float = PLANE_DRAW_NEAR,
                         d_far:  float = FOV_DEPTH_MAX,
                         lat_half: float = FOV_LAT_HALF) -> np.ndarray:
    """
    Build 4 corners of a ground-aligned rectangle that spans the full
    depth range [d_near, d_far] and lateral width [-lat_half, +lat_half].

    The rectangle lies ON the fitted plane (n·x = d), so every corner's
    height is solved from the plane equation.  This eliminates the blind
    spot between the rover and the far detection zone.

    Returns (4, 3) array in the camera frame, ordered for polyline drawing:
      [near-left, near-right, far-right, far-left]
    """
    up_col = 3 - depth_col - lat_col   # the remaining axis (0+1+2=3)
    denom = n[up_col] if abs(n[up_col]) > 1e-6 else 1e-6

    corners = []
    for dep, lat in [(d_near, -lat_half),
                     (d_near, +lat_half),
                     (d_far,  +lat_half),
                     (d_far,  -lat_half)]:
        p = np.zeros(3)
        p[depth_col] = dep
        p[lat_col]   = lat
        p[up_col]    = (d - n[depth_col] * dep - n[lat_col] * lat) / denom
        corners.append(p)
    return np.array(corners)


def decode_image(msg: Image) -> np.ndarray | None:
    """Convert a ROS Image (rgb8 / bgr8) to a BGR OpenCV Mat."""
    h, w = msg.height, msg.width
    enc  = msg.encoding.lower()
    arr  = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ("rgb8", "bgr8", "8uc3"):
        img = arr.reshape(h, w, 3)
        return img[..., ::-1].copy() if enc == "rgb8" else img.copy()
    if enc in ("rgba8", "bgra8"):
        img = arr.reshape(h, w, 4)
        img = img[..., :3]
        return img[..., ::-1].copy() if enc == "rgba8" else img.copy()
    if enc in ("mono8", "8uc1"):
        img = arr.reshape(h, w)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return None


# ──────────────────────────── ROS2 Node ───────────────────────────────────

class PlaneVisualizerNode(Node):

    def __init__(self, use_sdk=False):
        super().__init__("plane_visualizer")
        self._lock   = threading.Lock()
        self._latest = None       # dict with plane/points data
        self._profile_stats = None
        self._profile_segments = []
        self._profile_stats_history = deque(maxlen=PROFILE_STATS_HISTORY)
        self._profile_missing_frames = 0
        self._profile_last_stats = None
        self._img    = None       # latest BGR OpenCV image
        self._K      = None       # 3×3 camera intrinsic matrix
        self._new    = threading.Event()
        # ROS cloud analysis runs independently from image callbacks. Keep only
        # one pending message so a slow processor never builds a stale backlog.
        self._cloud_pending_lock = threading.Lock()
        self._cloud_pending = None
        self._cloud_worker_stop = threading.Event()
        self._cloud_worker = None

        self._normal_buf = deque(maxlen=NORMAL_BUF_SIZE)
        self._frame_count = 0
        self._haz_hist = deque(maxlen=HAZ_HIST)   # recent hazard-scan results
        self._haz_skip = 0                        # frames since the scan last ran

        # ── Overlay publisher (rover → base, replaces X11 forwarding) ───────
        # QoS: depth=1, best-effort – we only care about the latest frame.
        pub_qos = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
        )
        self.pub_overlay = self.create_publisher(Image, PUB_TOPIC, pub_qos)

        if use_sdk:
            self.get_logger().info("SDK mode: reading directly from ZED SDK")
            return

        # Keep only the newest sensor sample. A deeper subscription queue can
        # make the viewer render old images after a CPU-heavy frame.
        qos = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
        )

        self.sub_cloud = self.create_subscription(
            PointCloud2, "/zed/zed_node/point_cloud/cloud_registered",
            self._cloud_cb, qos)
        self.sub_img   = self.create_subscription(
            Image, "/zed/zed_node/rgb/color/rect/image",
            self._img_cb, qos)
        self.sub_info  = self.create_subscription(
            CameraInfo, "/zed/zed_node/rgb/color/rect/camera_info",
            self._info_cb, qos)

        self.get_logger().info(
            f"Subscribed to RGB image + camera_info + point_cloud. Waiting…"
            f"  [publishing overlay → {PUB_TOPIC} at {PUB_RESIZE[0]}×{PUB_RESIZE[1]}]"
        )

    # ── Overlay publish helper ───────────────────────────────────────────────
    def publish_overlay(self, bgr: np.ndarray):
        """Convert a BGR numpy image to sensor_msgs/Image and publish it."""
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "overlay"
        msg.height, msg.width = bgr.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = bgr.shape[1] * 3
        msg.data = bgr.tobytes()
        self.pub_overlay.publish(msg)

    # ── camera intrinsics ────────────────────────────────────────────────
    def _info_cb(self, msg: CameraInfo):
        if self._K is None:
            K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            with self._lock:
                self._K = K
            self.get_logger().info(
                f"Got camera intrinsics  fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
                f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}"
            )

    # ── RGB image ─────────────────────────────────────────────────────────
    def _img_cb(self, msg: Image):
        img = decode_image(msg)
        if img is None:
            self.get_logger().warning(
                f"Cannot decode image encoding '{msg.encoding}'"
            )
            return
        with self._lock:
            self._img = img
        self._new.set()

    # ── hazard persistence (3 of last 5 frames) ──────────────────────────
    def _update_hazard(self, res):
        if res is SKIP:
            self._haz_skip += 1
            if self._haz_skip > HAZ_HOLD_FRAMES:
                self._haz_hist.clear()
        else:
            self._haz_skip = 0
            self._haz_hist.append(res)

        hits = [r for r in self._haz_hist if r is not None]
        if len(hits) < HAZ_CONFIRM:
            return None
        latest = hits[-1]
        same = [r for r in hits if r["type"] == latest["type"]]
        return {
            "type":   latest["type"],
            "depth":  float(np.median([r["depth"] for r in same])),
            "lat_lo": latest["lat_lo"],
            "lat_hi": latest["lat_hi"],
            "why":    latest["why"],
        }

    # ── point cloud (ROS) ─────────────────────────────────────────────────
    def _cloud_cb(self, msg: PointCloud2):
        # PointCloud2 conversion and plane/hazard analysis are expensive. The
        # subscription callback must stay short so RGB callbacks remain fresh.
        with self._cloud_pending_lock:
            self._cloud_pending = msg
        self._new.set()

    def start_cloud_worker(self):
        if self._cloud_worker is not None:
            return
        self._cloud_worker = threading.Thread(
            target=self._cloud_worker_loop, name="point-cloud-analysis", daemon=True
        )
        self._cloud_worker.start()

    def _cloud_worker_loop(self):
        while not self._cloud_worker_stop.is_set():
            with self._cloud_pending_lock:
                msg, self._cloud_pending = self._cloud_pending, None
            if msg is None:
                self._cloud_worker_stop.wait(0.005)
                continue
            try:
                step_f    = msg.point_step // 4
                total_pts = msg.width * msg.height
                raw = np.frombuffer(msg.data, dtype=np.float32)
                raw = raw[: total_pts * step_f].reshape(-1, step_f)
                xyz = raw[:, :3].copy()
                valid    = np.all(np.isfinite(xyz), axis=1)
                pts_raw  = xyz[valid]
            except Exception as e:
                self.get_logger().error(f"Point cloud parse error: {e}")
                continue
            self._process_cloud(pts_raw, msg.header.frame_id)

    def _publish_profile_update(self, profile_result):
        """Publish profile values using the sole temporal filter: median stats."""
        global LAST_SLOPE_PROFILE
        raw_stats = profile_result["stats"] if profile_result is not None else None
        segments = profile_result["segments"] if profile_result is not None else []
        stats = None
        if raw_stats is not None:
            self._profile_stats_history.append({
                key: float(raw_stats[key]) for key in
                ("min", "avg", "max", "raw_min", "raw_avg", "raw_max")
            })
            stats = {
                key: float(np.median([sample[key] for sample in self._profile_stats_history]))
                for key in ("min", "avg", "max", "raw_min", "raw_avg", "raw_max")
            }
            stats["held"] = False
            self._profile_last_stats = stats
            self._profile_missing_frames = 0
        else:
            self._profile_missing_frames += 1
            if (self._profile_last_stats is not None and
                    self._profile_missing_frames <= PROFILE_STATS_HOLD_FRAMES):
                stats = dict(self._profile_last_stats)
                stats["held"] = True
            elif self._profile_missing_frames > PROFILE_STATS_HOLD_FRAMES:
                self._profile_stats_history.clear()
                self._profile_last_stats = None
        LAST_SLOPE_PROFILE = profile_result
        with self._lock:
            self._profile_stats = stats
            self._profile_segments = segments
            if self._latest is not None:
                updated = dict(self._latest)
                updated["slope_stats"] = stats
                updated["profile_segments"] = segments
                updated["baseline_deg"] = (profile_result.get("baseline_deg", 0.0)
                                           if profile_result is not None else updated.get("baseline_deg", 0.0))
                self._latest = updated
        return stats, segments

    def _mark_plane_stale(self, reason: str):
        """Hide rejected plane geometry while retaining confirmed hazards briefly."""
        held_hazard = self._update_hazard(SKIP)
        with self._lock:
            if self._latest is not None:
                updated = dict(self._latest)
                updated["plane_stale"] = True
                updated["hazard"] = held_hazard
                updated["haz_note"] = reason
                self._latest = updated
        self._new.set()

    # ── shared processing (ROS cloud or ZED SDK cloud) ───────────────────
    def _process_cloud(self, pts_raw: np.ndarray, frame: str):
        if pts_raw.shape[0] < 100:
            self._publish_profile_update(None)
            self._mark_plane_stale("too few cloud points")
            return

        # One-shot frame debug
        if not hasattr(self, "_dbg"):
            self._dbg = True
            mn, mx = pts_raw.min(0), pts_raw.max(0)
            self.get_logger().info(
                f"[FRAME DEBUG] x=[{mn[0]:.2f},{mx[0]:.2f}] "
                f"y=[{mn[1]:.2f},{mx[1]:.2f}] "
                f"z=[{mn[2]:.2f},{mx[2]:.2f}] "
                f"frame='{frame}'"
            )

        # Frame detection: optical → Z depth; else → X depth
        if "optical" in frame:
            depth_col, lat_col = 2, 0
            up_col, up_sign    = 1, -1.0   # optical: Y is down, so -Y is up
        else:
            depth_col, lat_col = 0, 1      # zed_left_camera_frame
            up_col, up_sign    = 2,  1.0   # Z is up

        # ── STAGE 1: slope ──────────────────────────────────────────────
        # (a) FOV filter: depth/lateral corridor plus a height gate centered on
        #     the learned floor line in camera coordinates. It opens with range
        #     to admit ramps/descents while lowest-surface sampling rejects clutter.
        # (b) select_ground_points(): keep only the LOWEST surface per
        #     (depth, lateral) cell, so furniture/walls above the floor are
        #     discarded.
        ht  = pts_raw[:, up_col] * up_sign          # height relative to camera, up = +
        dep = pts_raw[:, depth_col]
        # The slope allowance starts at the far-zone edge, where the global
        # plane fit begins, rather than opening fully at every range.
        ramp_distance = np.maximum(dep - FOV_DEPTH_MIN, 0.0)
        rise = np.minimum(ramp_distance * np.tan(np.radians(GATE_RAMP_DEG)), GATE_MAX_RISE)
        drop = np.minimum(ramp_distance * np.tan(np.radians(GATE_DESC_DEG)), GATE_MAX_DROP)

        # Profile candidates use only a loose absolute range gate. The actual
        # near-to-far continuity walk below rejects abrupt objects and clutter.
        profile_mask = (
            (dep >= PLANE_DRAW_NEAR) &
            (dep < FOV_DEPTH_MAX) &
            (pts_raw[:, lat_col] > -FOV_LAT_HALF) &
            (pts_raw[:, lat_col] < FOV_LAT_HALF) &
            (ht > -CAM_HEIGHT - PROFILE_ABS_BELOW) &
            (ht < -CAM_HEIGHT + PROFILE_ABS_ABOVE)
        )
        profile_candidates = pts_raw[profile_mask]
        profile_ground = select_ground_points(
            profile_candidates, depth_col, lat_col, up_col, up_sign
        )
        profile_result = plane_profile_slopes(
            profile_ground, depth_col, up_col, up_sign
        )
        profile_stats, profile_segments = self._publish_profile_update(profile_result)

        # In the camera frame, a flat floor can rise/fall with depth when the
        # camera is pitched. Center the ROI gate on the learned near-ground line
        # instead of a constant -CAM_HEIGHT; then allow the configured ramp/drop
        # envelope around that line.
        if profile_result.get("baseline_valid", False):
            ref_grade = np.tan(np.radians(profile_result["baseline_deg"]))
            expected_ground_h = profile_result["baseline_intercept"] + ref_grade * dep
        else:
            expected_ground_h = np.full_like(dep, -CAM_HEIGHT, dtype=float)
        z_max = expected_ground_h + GROUND_TOL_ABOVE + rise
        z_min = expected_ground_h - GROUND_TOL_BELOW - drop

        mask = (
            (dep >= FOV_DEPTH_MIN) &
            (dep <  FOV_DEPTH_MAX) &
            (pts_raw[:, lat_col] > -FOV_LAT_HALF) &
            (pts_raw[:, lat_col] <  FOV_LAT_HALF) &
            (ht > z_min) &
            (ht < z_max)
        )
        pts = pts_raw[mask]
        if pts.shape[0] < 30:
            self._mark_plane_stale("too few points in slope ROI")
            return

        # 1. (optional, legacy) dominant depth - OFF by default, see USE_DOMINANT_DEPTH
        if USE_DOMINANT_DEPTH:
            pts_dom = pts[find_dominant_depth(pts, depth_col)]
        else:
            pts_dom = pts

        # 2. lowest-surface ground sampling
        sampled_pts = select_ground_points(pts_dom, depth_col, lat_col, up_col, up_sign)

        # 3. RANSAC (skipped if there are too few ground cells)
        if sampled_pts.shape[0] >= GROUND_MIN_CELLS:
            ransac_result = ransac_plane(sampled_pts, depth_col)
        else:
            ransac_result = None

        if ransac_result is not None:
            n, d, centroid, planarity, inlier_mask = ransac_result
            # ── Plausibility check 1: reject wall/vertical fits ────────────
            n_up  = abs(n[up_col])
            n_fwd = abs(n[depth_col])
            if n_up < n_fwd:
                self._mark_plane_stale("wall-like fit rejected")
                return
            # ── Plausibility check 2: the plane must be BELOW the camera ───
            # n points up, so the camera origin's height above the plane is -d.
            # A plane fitted through chair seats / walls sits near or above
            # the camera and is rejected here.
            if -d < GROUND_MIN_CAM_FRAC * CAM_HEIGHT:
                self._mark_plane_stale("fit is not below camera")
                return

            # RANSAC only proves that points share a plane; it does not prove
            # that the plane is the floor. Require its inliers to agree with
            # the profile that was anchored near the camera and walked outward
            # through continuous ground-height bins.
            accepted_bins = profile_result.get("accepted_bins", {})
            inlier_pts = sampled_pts[inlier_mask]
            inlier_depth = inlier_pts[:, depth_col]
            inlier_height = inlier_pts[:, up_col] * up_sign
            inlier_bins = np.floor(inlier_depth / PROFILE_CELL_D).astype(int)
            matched_bins = {}
            for bin_idx, point_height in zip(inlier_bins, inlier_height):
                expected_height = accepted_bins.get(int(bin_idx))
                if expected_height is None:
                    continue
                matched_bins.setdefault(int(bin_idx), []).append(
                    abs(float(point_height) - float(expected_height)))
            bin_errors = [float(np.median(errors)) for errors in matched_bins.values()]
            if (len(bin_errors) < GROUND_PROFILE_MATCH_BINS or
                    float(np.median(bin_errors)) > GROUND_PROFILE_MATCH_TOL):
                self._mark_plane_stale("plane fit does not match ground profile")
                return
            plane_ok = True
        else:
            # Three arbitrary points always define a plane, even when they are
            # a wall edge or unrelated returns. Fail closed instead of drawing
            # that unsupported plane as if it were ground.
            self._mark_plane_stale("no supported ground-plane fit")
            return

        # Confidence is decided right after the fit, so it also gates smoothing.
        inlier_ratio = np.sum(inlier_mask) / max(1, len(inlier_mask))
        confident = (plane_ok and planarity >= MIN_PLANARITY
                     and inlier_ratio >= MIN_INLIER_RATIO)

        # smoothing buffer: only CONFIDENT planes go in, so a weak or bad frame
        # cannot contaminate the next 10 frames of the average
        if confident:
            self._normal_buf.append(n)
        if len(self._normal_buf) > 0:
            smoothed_n = np.mean(self._normal_buf, axis=0)
            smoothed_n /= np.linalg.norm(smoothed_n)
        else:
            smoothed_n = n

        slope = compute_slope(smoothed_n, depth_col)

        # Current-frame spatial stats: grades of local depth segments from the
        # near edge to the far edge of the drawn plane.
        slope_stats = profile_stats

        # d_ref must come first – used by both ext_corners and the hazard scan
        d_ref = float(np.dot(smoothed_n, centroid))

        # ── Extended plane rectangle (covers PLANE_DRAW_NEAR → FOV_DEPTH_MAX) ─
        # For confident frames use the full-corridor rect; fallback keeps the
        # small square so something is always visible.
        if plane_ok:
            ext_corners = ground_plane_rect_3d(
                smoothed_n, d_ref, depth_col, lat_col
            )
        else:
            ext_corners = plane_corners_3d(centroid[None, :], smoothed_n, PLANE_HALF_SIZE)

        # Legacy small square kept in candidate for backward compat (not drawn)
        corners = plane_corners_3d(centroid[None, :], smoothed_n, PLANE_HALF_SIZE)

        # ── Near-zone display points (PLANE_DRAW_NEAR → FOV_DEPTH_MIN) ───
        # Display-only: fills the bottom of the plane with green dots.
        # Uses a LIGHTER grid (min 2 pts/cell, looser plane tolerance) so
        # sparse close-range ground returns still produce visible dots.
        near_pts = np.empty((0, 3))
        if confident:
            near_mask = (
                (dep >= PLANE_DRAW_NEAR) &
                (dep <  FOV_DEPTH_MIN) &
                (pts_raw[:, lat_col] > -FOV_LAT_HALF) &
                (pts_raw[:, lat_col] <  FOV_LAT_HALF) &
                (ht > z_min) &
                (ht < z_max)
            )
            near_raw = pts_raw[near_mask]
            if near_raw.shape[0] > 5:
                # Light grid sampler: 1 point per (GROUND_CELL_D × GROUND_CELL_L)
                # cell with min 2 pts required (half the far-zone requirement).
                n_dep = near_raw[:, depth_col]
                n_lat = near_raw[:, lat_col]
                n_ht  = near_raw[:, up_col] * up_sign
                dep_min_n = float(n_dep.min()); dep_max_n = float(n_dep.max())
                lat_min_n = float(n_lat.min()); lat_max_n = float(n_lat.max())
                nd_n = max(1, int(np.ceil((dep_max_n - dep_min_n) / GROUND_CELL_D)))
                nl_n = max(1, int(np.ceil((lat_max_n - lat_min_n) / GROUND_CELL_L)))
                di_n = np.clip(((n_dep - dep_min_n) / GROUND_CELL_D).astype(int), 0, nd_n - 1)
                li_n = np.clip(((n_lat - lat_min_n) / GROUND_CELL_L).astype(int), 0, nl_n - 1)
                cell_n = di_n * nl_n + li_n

                order_n = np.argsort(cell_n, kind="stable")
                cs_n = cell_n[order_n]
                _, first_n = np.unique(cs_n, return_index=True)
                last_n = np.append(first_n[1:], cs_n.size)

                near_gnd = []
                for a, b in zip(first_n, last_n):
                    if b - a < 2:           # only 2 pts/cell required (near is sparse)
                        continue
                    idx_n = order_n[a:b]
                    hh = n_ht[idx_n]
                    floor_h = np.percentile(hh, GROUND_PERCENTILE)
                    low_n = idx_n[hh <= floor_h + GROUND_BAND]
                    if low_n.size == 0:
                        continue
                    hl_n = n_ht[low_n]
                    near_gnd.append(near_raw[low_n[np.argmin(np.abs(hl_n - np.median(hl_n)))]])

                if near_gnd:
                    ng = np.array(near_gnd)
                    # loose plane-distance filter (8× RANSAC_THRESH ≈ 0.24 m)
                    plane_dist = np.abs(ng @ smoothed_n - d_ref)
                    near_pts = ng[plane_dist < RANSAC_THRESH * 8]

        # ── STAGE 2: hazard scan ─────────────────────────────────────────
        note = None
        if confident and slope["abs_theta"] < HAZ_REF_MAX_TILT:
            res, note = analyze_hazards(pts_raw, smoothed_n, d_ref, depth_col, lat_col)
        else:
            res = SKIP
        hazard = self._update_hazard(res)

        candidate = {
            "n": smoothed_n,
            "d": d_ref,
            "corners": ext_corners,       # extended rect – used for drawing
            "corners_small": corners,     # small centroid square – kept for compat
            "frame": frame,
            "slope": slope,
            "slope_stats": slope_stats,
            "profile_segments": profile_segments,
            "baseline_deg": profile_result.get("baseline_deg", 0.0),
            "plane_stale": False,
            "sampled_pts": sampled_pts,
            "inlier_mask": inlier_mask,
            "near_pts": near_pts,         # near-zone display-only dots
            "planarity": planarity,
            "centroid": centroid,
            "hazard": hazard,
            "haz_note": note,
            "depth_col": depth_col,
            "lat_col": lat_col,
        }

        with self._lock:
            self._latest = candidate
            self._frame_count += 1
            candidate["frame_count"] = self._frame_count

        self._new.set()


# ──────────────────────────── ZED SDK source ──────────────────────────────

class ZedSDKSource(threading.Thread):
    """Grabs image + XYZ cloud straight from the ZED SDK and feeds the node."""

    def __init__(self, node: PlaneVisualizerNode):
        super().__init__(daemon=True)
        self.node = node
        self.zed = sl.Camera()
        self.running = True
        self.auto = SDK_AUTO_EXPOSURE
        self.exposure = SDK_EXPOSURE
        self.gain = SDK_GAIN
        self._pending = None
        self._plock = threading.Lock()

    def open(self):
        init = sl.InitParameters()
        init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD  # = zed_left_camera_frame
        init.coordinate_units = sl.UNIT.METER
        # depth_mode intentionally left at the SDK default
        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(
                f"ZED open failed: {err}  (is the ROS zed wrapper still running?)")

        cal = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
        K = np.array([[cal.fx, 0, cal.cx],
                      [0, cal.fy, cal.cy],
                      [0, 0, 1]], dtype=np.float64)
        with self.node._lock:
            self.node._K = K
        self._apply()

    def _apply(self):
        z = self.zed
        if self.auto:
            z.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 1)
        else:
            z.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)
            z.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, self.exposure)
            z.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, self.gain)
        print(f"[zed] auto={self.auto} exposure={self.exposure} gain={self.gain}")

    def request(self, d_exp=0, d_gain=0, toggle_auto=False):
        """Called from the UI thread; applied in the grab thread."""
        with self._plock:
            e, g, t = self._pending or (0, 0, False)
            self._pending = (e + d_exp, g + d_gain, t or toggle_auto)

    def _apply_pending(self):
        with self._plock:
            p, self._pending = self._pending, None
        if p is None:
            return
        d_exp, d_gain, toggle = p
        self.exposure = int(np.clip(self.exposure + d_exp, 0, 100))
        self.gain = int(np.clip(self.gain + d_gain, 0, 100))
        if toggle:
            self.auto = not self.auto
        self._apply()

    def run(self):
        rt = sl.RuntimeParameters()
        img_mat, cloud = sl.Mat(), sl.Mat()
        res = sl.Resolution(SDK_CLOUD_W, SDK_CLOUD_H)
        while self.running:
            self._apply_pending()
            if self.zed.grab(rt) != sl.ERROR_CODE.SUCCESS:
                continue

            self.zed.retrieve_image(img_mat, sl.VIEW.LEFT)
            bgr = np.ascontiguousarray(img_mat.get_data()[..., :3])   # BGRA -> BGR
            with self.node._lock:
                self.node._img = bgr

            self.zed.retrieve_measure(cloud, sl.MEASURE.XYZ, sl.MEM.CPU, res)
            xyz = cloud.get_data()[..., :3].reshape(-1, 3)
            xyz = xyz[np.isfinite(xyz).all(axis=1)]
            self.node._process_cloud(xyz, "zed_left_camera_frame")
            self.node._new.set()

    def stop(self):
        self.running = False
        self.join(timeout=2)
        self.zed.close()


# ──────────────────────────── OpenCV Renderer ─────────────────────────────

def _draw_rect_alpha(img, x1, y1, x2, y2, colour, alpha):
    """Draw a filled semi-transparent rectangle, blending only its ROI."""
    h, w = img.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x1 >= x2 or y1 >= y2:
        return
    roi = img[y1:y2, x1:x2]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (x2 - x1 - 1, y2 - y1 - 1), colour, -1)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)

def _label_val(img, label, value, x, y, lw=1, label_col=(140, 140, 160), val_col=(220, 220, 220)):
    """Draw a left-aligned label:value pair."""
    cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, label_col, lw, cv2.LINE_AA)
    lbl_w, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, lw)[0]
    cv2.putText(img, value, (x + lbl_w + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, val_col, lw, cv2.LINE_AA)


def draw_overlay(img: np.ndarray, data: dict, K: np.ndarray) -> np.ndarray:
    """Render AR overlay on img and return it."""
    out    = img.copy()
    h, w   = out.shape[:2]

    n = data["n"]
    d = data["d"]
    corners = data["corners"]
    slope = data["slope"]
    sampled_pts = data["sampled_pts"]
    inlier_mask = data["inlier_mask"]
    planarity = data["planarity"]
    fc = data.get("frame_count", 0)
    plane_stale = data.get("plane_stale", False)
    # Plane geometry is hidden while stale, but preserve the last confirmed
    # hazard during its configured hold window.
    hz = data.get("hazard")
    note = None if plane_stale else data.get("haz_note")

    inlier_ratio = np.sum(inlier_mask) / max(1, len(inlier_mask))
    is_confident = (not plane_stale and planarity >= MIN_PLANARITY
                    and inlier_ratio >= MIN_INLIER_RATIO)
    col_plane = slope["colour"] if is_confident else (0, 140, 255)

    # ── plane polygon ────────────────────────────────────────────────────────
    if is_confident:
        corner_px = project_pts(corners, K)
        if corner_px is not None:
            overlay = out.copy()
            cv2.fillPoly(overlay, [corner_px.reshape(-1, 1, 2)], col_plane)
            cv2.addWeighted(overlay, 0.28, out, 0.72, 0, out)
            cv2.polylines(out, [corner_px.reshape(-1, 1, 2)], True, col_plane, 2, cv2.LINE_AA)

    # ── hazard band on the ground ───────────────────────────────────────────
    hz_col = None
    if hz is not None:
        hz_col = COL_HAZ_OBST if hz["type"] == "OBSTACLE" else COL_HAZ_PIT
        band3d = hazard_band_3d(hz, n, d, data["depth_col"], data["lat_col"])
        band_px = project_pts(band3d, K)
        if band_px is not None:
            overlay = out.copy()
            cv2.fillPoly(overlay, [band_px.reshape(-1, 1, 2)], hz_col)
            cv2.addWeighted(overlay, 0.40, out, 0.60, 0, out)
            cv2.polylines(out, [band_px.reshape(-1, 1, 2)], True, hz_col, 3, cv2.LINE_AA)

    # ── inlier dots (and, optionally, small dim outlier dots) ───────────────
    pxs = None if plane_stale else project_pts(sampled_pts, K)
    if pxs is not None:
        for i, px in enumerate(pxs):
            u, v = int(px[0]), int(px[1])
            if not (0 <= u < w and 0 <= v < h):
                continue
            ok = inlier_mask[i] if i < len(inlier_mask) else True
            if ok:
                cv2.circle(out, (u, v), 5, (0, 230, 80), -1, cv2.LINE_AA)
                cv2.circle(out, (u, v), 5, (0, 100, 40), 1, cv2.LINE_AA)
            elif SHOW_OUTLIERS:
                cv2.circle(out, (u, v), 2, (40, 40, 160), -1, cv2.LINE_AA)

    # ── near-zone dots (display only, slightly smaller & dimmer green) ──────
    near_pts = None if plane_stale else data.get("near_pts")
    if near_pts is not None and len(near_pts) > 0:
        npx = project_pts(near_pts, K)
        if npx is not None:
            for px in npx:
                u, v = int(px[0]), int(px[1])
                if 0 <= u < w and 0 <= v < h:
                    cv2.circle(out, (u, v), 4, (0, 200, 60), -1, cv2.LINE_AA)
                    cv2.circle(out, (u, v), 4, (0, 90, 30), 1, cv2.LINE_AA)

    # ── flashing red border (WALL slope, or hazard closer than HAZ_FLASH_DIST) ─
    if is_confident and slope["classification"] == "WALL / OBSTACLE" and (fc % 4) < 2:
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
    if hz is not None and hz["depth"] <= HAZ_FLASH_DIST and (plane_stale or (fc % 4) < 2):
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), hz_col, 8)

    # ════════════════════════════════════════════════════════════════════════
    # LEFT  HUD  PANEL  — compact info card
    # ════════════════════════════════════════════════════════════════════════
    PANEL_W = 280
    PANEL_X = 10
    PANEL_Y = 10
    LH = 14          # line height px
    PAD = 6          # inner padding
    FS  = 0.32       # font scale

    # ── Title bar ────────────────────────────────────────────────────────────
    title_h = 20
    _draw_rect_alpha(out, PANEL_X, PANEL_Y, PANEL_X + PANEL_W, PANEL_Y + title_h, (30, 10, 60), 0.82)
    cv2.putText(out, "ZED2i  PLANE VISUALIZER",
                (PANEL_X + PAD, PANEL_Y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 140, 255), 1, cv2.LINE_AA)
    cv2.line(out, (PANEL_X, PANEL_Y + title_h), (PANEL_X + PANEL_W, PANEL_Y + title_h), (140, 80, 200), 1)

    # ── Body rows ────────────────────────────────────────────────────────────
    haz_txt = "clear" if hz is None else f"{hz['type']} {hz['depth']:.1f}m"
    why_txt = hz["why"] if hz is not None else (note or "-")
    rows = [
        ("PLANE:",    "stale" if plane_stale else "current"),
        ("N:",        f"{n[0]:+.2f}  {n[1]:+.2f}  {n[2]:+.2f}"),
        ("PITCH:",    f"{slope['pitch']:+.1f} deg"),
        ("CAMBER:",   f"{slope['camber']:+.1f} deg"),
        ("CAM H:",    f"{-d:.2f} m above plane  (set {CAM_HEIGHT:.2f})"),
        ("PLANARITY:",f"{planarity:.2f}  |  {int(np.sum(inlier_mask))}/{len(inlier_mask)}"),
        ("HAZARD:",   haz_txt),
        ("WHY:",      why_txt),
    ]

    body_h = len(rows) * LH + PAD * 2 + 6
    _draw_rect_alpha(out, PANEL_X, PANEL_Y + title_h, PANEL_X + PANEL_W, PANEL_Y + title_h + body_h, (10, 8, 20), 0.72)

    for i, (lbl, val) in enumerate(rows):
        y = PANEL_Y + title_h + PAD + (i + 1) * LH
        vc = (190, 190, 200) if not (lbl in ("HAZARD:", "WHY:") and hz is not None) else hz_col
        _label_val(out, lbl, val, PANEL_X + PAD, y, lw=1,
                   label_col=(110, 110, 130), val_col=vc)

    cv2.rectangle(out, (PANEL_X, PANEL_Y),
                  (PANEL_X + PANEL_W, PANEL_Y + title_h + body_h),
                  (100, 60, 160), 1)

    # ── Hazard banner (top centre) ───────────────────────────────────────────
    if hz is not None:
        txt = f"{hz['type']} AHEAD  {hz['depth']:.1f} m"
        (bw_t, bh_t), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        bx = max(PANEL_X + PANEL_W + 12, (w - bw_t) // 2 - 14)
        by = 12
        _draw_rect_alpha(out, bx, by, bx + bw_t + 28, by + bh_t + 22, hz_col, 0.85)
        cv2.putText(out, txt, (bx + 14, by + bh_t + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # ════════════════════════════════════════════════════════════════════════
    # BOTTOM  STATUS  BAR
    # ════════════════════════════════════════════════════════════════════════
    BAR_H = 78
    bar_y = h - BAR_H
    _draw_rect_alpha(out, 0, bar_y, w, h, (10, 8, 20), 0.80)
    cv2.line(out, (0, bar_y), (w, bar_y), col_plane, 2)   # colored top border

    # ── MIN / AVG / MAX slope readout (left side of bar) ────────────────
    ss      = data.get("slope_stats")
    s_min   = ss["min"] if ss is not None else None
    s_avg   = ss["avg"] if ss is not None else None
    s_max   = ss["max"] if ss is not None else None

    LBL_COL = (130, 120, 150)
    VAL_FS  = 0.70
    VAL_TH  = 2

    row_ys = [bar_y + 20, bar_y + 44, bar_y + 68]
    held_label = " (held)" if ss is not None and ss.get("held", False) else ""
    row_data = [
        ("MIN", (f"{s_min:+.1f} deg" if s_min is not None else "--") + held_label),
        ("AVG", (f"{s_avg:+.1f} deg" if s_avg is not None else "--") + held_label),
        ("MAX", (f"{s_max:+.1f} deg" if s_max is not None else "--") + held_label),
    ]

    max_val_w = 0
    x_lbl = 14
    for (lbl, val), ry in zip(row_data, row_ys):
        # dim label
        (lw_, _), _ = cv2.getTextSize(lbl + " ", cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
        cv2.putText(out, lbl, (x_lbl, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, LBL_COL, 1, cv2.LINE_AA)
        # bright value
        x_val = x_lbl + lw_ + 2
        cv2.putText(out, val, (x_val, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, VAL_FS, col_plane, VAL_TH, cv2.LINE_AA)
        (vw_, _), _ = cv2.getTextSize(val, cv2.FONT_HERSHEY_SIMPLEX, VAL_FS, VAL_TH)
        max_val_w = max(max_val_w, x_val + vw_)

    div_x = max_val_w + 18
    cv2.line(out, (div_x, bar_y + 6), (div_x, h - 6), (70, 60, 90), 1)

    if is_confident:
        cls = slope["classification"]
        cv2.putText(out, cls, (div_x + 20, bar_y + BAR_H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, col_plane, 2, cv2.LINE_AA)
    else:
        msg = "UNCERTAIN - GLOBAL PLANE FIT"
        cv2.putText(out, msg, (div_x + 20, bar_y + BAR_H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 255), 2, cv2.LINE_AA)

    bar_rx = w - 160
    bar_rw = 130
    bar_rh = 12
    bar_ry = bar_y + (BAR_H - bar_rh) // 2
    cv2.rectangle(out, (bar_rx, bar_ry), (bar_rx + bar_rw, bar_ry + bar_rh), (50, 40, 60), -1)
    if s_min is not None and s_max is not None:
        severity = max(abs(s_min), abs(s_max))
        fill_w = int(bar_rw * min(1.0, severity / 90.0))
        if fill_w > 0:
            cv2.rectangle(out, (bar_rx, bar_ry), (bar_rx + fill_w, bar_ry + bar_rh), col_plane, -1)
    cv2.rectangle(out, (bar_rx, bar_ry), (bar_rx + bar_rw, bar_ry + bar_rh), (100, 80, 120), 1)
    cv2.putText(out, "0         90", (bar_rx, bar_ry + bar_rh + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (100, 100, 110), 1, cv2.LINE_AA)
    cv2.putText(out, "SEVERITY", (bar_rx, bar_ry - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (120, 110, 140), 1, cv2.LINE_AA)

    return out


# ──────────────────────────── Main loop ───────────────────────────────────

WIN = "ZED2i Plane Visualizer"

def main():
    rclpy.init()
    node = PlaneVisualizerNode(use_sdk=USE_SDK)

    src = None
    if USE_SDK:
        if not HAVE_ZED_SDK:
            raise SystemExit("pyzed not installed - cannot use --sdk")
        src = ZedSDKSource(node)
        src.open()
        src.start()

    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True
    )
    spin_thread.start()

    WIN_W, WIN_H = 1280, 720
    node.start_cloud_worker()

    if HEADLESS:
        print(f"[headless] No local window. Publishing overlay → {PUB_TOPIC}  "
              f"({PUB_RESIZE[0]}×{PUB_RESIZE[1]})")
        print("Ctrl-C to quit.")
    else:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, WIN_W, WIN_H)
        print("Window open.  Q / Esc = quit | D = lane profile | "
              "SDK: 1/2 exposure -/+, 3/4 gain -/+, 0 auto on/off")

    while True:
        node._new.wait(timeout=0.05)
        node._new.clear()

        with node._lock:
            # _img is replaced atomically by the callback, never modified in
            # place. draw_overlay makes the single writable frame copy below.
            img  = node._img
            data = node._latest
            K    = node._K
            profile_stats = node._profile_stats

        # ── Build the frame to publish / display ─────────────────────────
        if img is None:
            frame = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
            cv2.putText(frame, "Waiting for RGB image...", (WIN_W // 4, WIN_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 180, 180), 2, cv2.LINE_AA)
        elif data is None or K is None:
            frame = img.copy()
            status_msg = "Waiting for camera_info..." if K is None else "Searching for plane..."
            cv2.putText(frame, status_msg, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2, cv2.LINE_AA)
            held_suffix = " (held)" if profile_stats is not None and profile_stats.get("held", False) else ""
            stats_text = (f"MIN / AVG / MAX: {profile_stats['min']:+.1f} / "
                          f"{profile_stats['avg']:+.1f} / {profile_stats['max']:+.1f} deg"
                          f"{held_suffix}"
                          if profile_stats is not None else "MIN / AVG / MAX: --")
            cv2.putText(frame, stats_text, (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 80), 2, cv2.LINE_AA)
        else:
            frame = draw_overlay(img, data, K)

        # ── Always publish (rover → base over ROS network) ───────────────
        # Overlay publication is only needed for the remote/headless viewer.
        if HEADLESS:
            node.publish_overlay(cv2.resize(frame, PUB_RESIZE))

        # ── Local display (only when not headless) ───────────────────────
        if not HEADLESS:
            # Display at the rendered/native resolution; avoid resizing every
            # frame to a large fixed window on the CPU.
            cv2.imshow(WIN, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("d"), ord("D")):
                print_lane_profile()
            if src is not None:
                if   key == ord("1"): src.request(d_exp=-SDK_STEP)
                elif key == ord("2"): src.request(d_exp=+SDK_STEP)
                elif key == ord("3"): src.request(d_gain=-SDK_STEP)
                elif key == ord("4"): src.request(d_gain=+SDK_STEP)
                elif key == ord("0"): src.request(toggle_auto=True)

    if src is not None:
        src.stop()
    node._cloud_worker_stop.set()
    if node._cloud_worker is not None:
        node._cloud_worker.join(timeout=2)
    if not HEADLESS:
        cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.try_shutdown()
    spin_thread.join(timeout=2)
    print("Exited cleanly.")


if __name__ == "__main__":
    main()
