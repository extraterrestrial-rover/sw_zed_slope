#!/usr/bin/env python3
"""Run deterministic synthetic point clouds through plane_visualizer hazard logic.

Usage: python3 mock_hazard_scenarios.py
The points use zed_left_camera_frame coordinates: X=depth, Y=lateral, Z=up.
"""

import numpy as np

from plane_visualizer import SKIP, analyze_hazards


DEPTH_COL = 0
LAT_COL = 1
GROUND_NORMAL = np.array([0.0, 0.0, 1.0])
GROUND_D = 0.0


def make_scene(kind):
    """Create a dense, three-lane cloud with one configurable terrain feature."""
    rng = np.random.default_rng(7)
    points = []
    strip_count = int(round((5.0 - 0.5) / 0.1))

    # [start, end) ranges are metres in front of the camera.
    features = {
        "flat": (None, None, None),
        "trench": (2.0, 2.6, -1.0),
        "narrow_trench": (2.0, 2.3, -1.0),
        "wide_pit": (1.8, 3.1, -1.2),
        "shallow_step": (2.2, 4.5, -0.35),
        "edge_gap": (2.0, 2.6, None),
        "end_gap": (4.3, 5.0, None),
        "ramp": (None, None, None),
        "obstacle": (2.0, 2.5, 0.45),
        "obstacle_then_gap": (2.0, 2.5, 0.45),
        "raised_platform_then_gap": (0.9, 1.5, 0.45),
        "one_lane_trench": (2.0, 2.6, -1.0),
        "noisy_ground": (None, None, None),
    }
    if kind not in features:
        raise ValueError(f"unknown scene: {kind}")
    feature_start, feature_end, feature_height = features[kind]

    for lane in range(3):
        lat_center = -0.5 + lane * 0.5
        for strip in range(strip_count):
            depth = 0.5 + (strip + 0.5) * 0.1
            height = 0.0
            in_feature = (feature_start is not None and
                          feature_start <= depth < feature_end)
            lane_applies = kind != "one_lane_trench" or lane == 1

            if kind == "ramp":
                height = min(0.25, max(0.0, depth - 1.0) * 0.08)
            elif kind == "shallow_step" and depth >= feature_start:
                height = feature_height
            elif kind == "raised_platform_then_gap" and depth >= feature_start:
                height = feature_height
                if depth >= feature_end:
                    continue
            elif kind in ("edge_gap", "end_gap", "obstacle_then_gap") and in_feature:
                continue
            elif kind == "trench" and in_feature:
                # Deep returns are retained for floor evidence but excluded from
                # the ordinary ground profile by HAZ_BELOW_MAX.
                height = feature_height
            elif kind in ("narrow_trench", "wide_pit", "one_lane_trench") and in_feature and lane_applies:
                height = feature_height
            elif kind in ("obstacle", "obstacle_then_gap") and in_feature:
                # Ground remains visible; add a raised, vertically spread object.
                points.extend(_patch(rng, depth, lat_center, 0.0, 30))
                points.extend(_patch(rng, depth, lat_center, feature_height, 30, spread=0.35))
                continue

            spread = 0.04 if kind == "noisy_ground" else 0.012
            points.extend(_patch(rng, depth, lat_center, height, 60, spread=spread))

    return np.asarray(points, dtype=np.float64)


def _patch(rng, depth, lateral, height, count, spread=0.012):
    return np.column_stack((
        depth + rng.uniform(-0.035, 0.035, count),
        lateral + rng.uniform(-0.18, 0.18, count),
        height + rng.normal(0.0, spread, count),
    ))


def main():
    cases = (
        ("flat ground", "flat", None),
        ("wide trench with floor", "trench", "PIT / TRENCH"),
        ("narrow trench (minimum gap width)", "narrow_trench", "PIT / TRENCH"),
        ("wide and deep pit", "wide_pit", "PIT / TRENCH"),
        ("shallow persistent step down", "shallow_step", "DROP-OFF"),
        ("continuous ramp", "ramp", None),
        ("raised obstacle", "obstacle", "OBSTACLE"),
        ("single-lane trench", "one_lane_trench", "PIT / TRENCH"),
        ("noisy flat ground", "noisy_ground", None),
        ("missing returns with no floor", "edge_gap", None),
        ("missing returns near scan limit", "end_gap", None),
        ("obstacle followed by missing returns", "obstacle_then_gap", None),
        ("raised platform followed by missing returns", "raised_platform_then_gap", None),
    )
    failed = False

    for label, scene, expected in cases:
        result, note = analyze_hazards(
            make_scene(scene), GROUND_NORMAL, GROUND_D, DEPTH_COL, LAT_COL
        )
        if result is SKIP:
            print(f"FAIL {label}: scan skipped (check mock cloud density/configuration)")
            failed = True
            continue

        actual = None if result is None else result["type"]
        # A bare no-return gap is intentionally reported as unverified in the
        # note rather than being promoted to a confirmed hazard.
        passed = actual == expected
        status = "PASS" if passed else "FAIL"
        detail = "clear" if result is None else (
            f"{result['type']} at {result['depth']:.1f} m ({result['why']})"
        )
        print(f"{status} {label}: {detail}" + (f"; note: {note}" if note else ""))
        failed |= not passed

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
