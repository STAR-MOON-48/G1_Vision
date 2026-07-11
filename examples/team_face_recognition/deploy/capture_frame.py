#!/usr/bin/env python3
"""Capture one ROS2 RGB frame for deployment diagnostics."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_face_bridge import image_message_to_bgr  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/camera/color/image_raw")
    parser.add_argument("--output", default="/tmp/g1_live_rgb.jpg")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("g1_rgb_frame_capture")
    qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    captured = False

    def on_image(message: Image) -> None:
        nonlocal captured
        image = image_message_to_bgr(message)
        if not cv2.imwrite(args.output, image):
            raise RuntimeError(f"Failed to write {args.output}")
        print(
            f"CAPTURE_OK path={args.output} width={message.width} "
            f"height={message.height} encoding={message.encoding}"
        )
        captured = True

    subscription = node.create_subscription(Image, args.topic, on_image, qos)
    deadline = time.monotonic() + args.timeout
    try:
        while not captured and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        if not captured:
            raise TimeoutError(f"No frame received from {args.topic}")
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
