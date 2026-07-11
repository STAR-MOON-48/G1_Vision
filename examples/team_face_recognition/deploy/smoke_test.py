#!/usr/bin/env python3
"""End-to-end smoke test for the running G1 face bridge.

Publishes synthetic RGB/depth frames over ROS2/DDS and verifies the JSON event
received on the Agent UDP endpoint. No biometric data is used by this test.
"""

from __future__ import annotations

import json
import time

import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String


RGB_TOPIC = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/aligned_depth_to_color/image_raw"
RESULT_TOPIC = "/ai/face_recognition/internal"


def image_message(node, *, encoding: str, width: int, height: int, data: bytes, step: int) -> Image:
    message = Image()
    message.header.stamp = node.get_clock().now().to_msg()
    message.header.frame_id = "deployment_smoke_test"
    message.height = height
    message.width = width
    message.encoding = encoding
    message.is_bigendian = 0
    message.step = step
    message.data = data
    return message


def main() -> None:
    rclpy.init()
    node = rclpy.create_node("g1_face_bridge_smoke_test")
    qos = QoSProfile(
        depth=5,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    rgb_publisher = node.create_publisher(Image, RGB_TOPIC, qos)
    depth_publisher = node.create_publisher(Image, DEPTH_TOPIC, qos)
    received: list[dict] = []

    def on_result(message: String) -> None:
        received.append(json.loads(message.data))

    result_subscription = node.create_subscription(String, RESULT_TOPIC, on_result, 10)

    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if rgb_publisher.get_subscription_count() and depth_publisher.get_subscription_count():
                break
        else:
            raise RuntimeError("Bridge subscriptions were not discovered")

        width = height = 64
        depth = image_message(
            node,
            encoding="16UC1",
            width=width,
            height=height,
            step=width * 2,
            data=(1500).to_bytes(2, "little") * (width * height),
        )
        rgb = image_message(
            node,
            encoding="bgr8",
            width=width,
            height=height,
            step=width * 3,
            data=bytes(width * height * 3),
        )
        rgb.header.stamp = depth.header.stamp

        depth_publisher.publish(depth)
        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.05)
        rgb_publisher.publish(rgb)

        deadline = time.monotonic() + 30.0
        while not received and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not received:
            raise TimeoutError("No face-recognition result received")
        payload = received[0]
        assert payload["event"] == "face_recognition"
        assert payload["event_id"]
        assert payload["session_id"]
        assert payload["source"]["rgb_topic"] == RGB_TOPIC
        assert payload["source"]["depth_topic"] == DEPTH_TOPIC
        assert isinstance(payload["faces"], list)
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        print("SMOKE_OK")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
