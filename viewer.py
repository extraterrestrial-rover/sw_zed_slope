#!/usr/bin/env python3
"""
viewer.py – Base-laptop viewer for the ZED2i slope/hazard overlay.

Run on the BASE laptop (same ROS2 domain as the rover):
    python3 viewer.py

The rover publishes the rendered overlay to /slope_visualizer/overlay.
This script subscribes to that topic and shows it in a local OpenCV window.
No X11 / SSH -X needed.

Key bindings
  Q / Esc  – quit
"""

import sys
import threading
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

TOPIC   = "/slope_visualizer/overlay"   # must match PUB_TOPIC in plane_visualizer.py
WIN     = "ZED2i Slope Viewer  [base]"
WIN_W   = 1280
WIN_H   = 720


class OverlayViewer(Node):
    def __init__(self):
        super().__init__("slope_overlay_viewer")
        self._lock  = threading.Lock()
        self._frame = None

        qos = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
        )
        self.sub = self.create_subscription(Image, TOPIC, self._cb, qos)
        self.get_logger().info(f"Subscribing to {TOPIC} …")

    def _cb(self, msg: Image):
        enc = msg.encoding.lower()
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        try:
            img = arr.reshape(msg.height, msg.width, 3)
        except ValueError:
            self.get_logger().warning("Unexpected image shape – skipping frame")
            return
        # Handle encoding
        if enc == "rgb8":
            img = img[..., ::-1].copy()
        else:
            img = img.copy()   # bgr8 – already correct
        with self._lock:
            self._frame = img

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None


def main():
    rclpy.init()
    node = OverlayViewer()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, WIN_W, WIN_H)

    print(f"Listening on {TOPIC} …  Q / Esc to quit.")

    while True:
        frame = node.get_frame()
        if frame is None:
            blank = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
            cv2.putText(blank, f"Waiting for {TOPIC} …",
                        (WIN_W // 8, WIN_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2, cv2.LINE_AA)
            cv2.imshow(WIN, blank)
        else:
            cv2.imshow(WIN, cv2.resize(frame, (WIN_W, WIN_H)))

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.try_shutdown()
    spin_thread.join(timeout=2)
    print("Viewer closed.")


if __name__ == "__main__":
    main()
