#!/usr/bin/env python3
"""ROS2/DDS bridge from G1 RGB/depth streams to face-recognition events.

The ROS imports are optional at module import time so the conversion and depth
helpers remain unit-testable on the Windows development machine.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import onnxruntime

from team_face_demo import FaceDatabase, build_gallery, create_face_app, face_bbox, match_embedding


try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    ROS_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # exercised on the Windows development host
    rclpy = None  # type: ignore[assignment]
    Node = object  # type: ignore[assignment,misc]
    qos_profile_sensor_data = None  # type: ignore[assignment]
    Image = Any  # type: ignore[assignment,misc]
    String = Any  # type: ignore[assignment,misc]
    ROS_IMPORT_ERROR = exc


SCHEMA_VERSION = 1


def stamp_to_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def image_message_to_bgr(message: Any) -> np.ndarray:
    encoding = str(message.encoding).lower()
    channels_by_encoding = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4, "mono8": 1}
    if encoding not in channels_by_encoding:
        raise ValueError(f"Unsupported RGB encoding: {message.encoding}")
    channels = channels_by_encoding[encoding]
    row_bytes = int(message.width) * channels
    buffer = np.frombuffer(message.data, dtype=np.uint8)
    expected = int(message.height) * int(message.step)
    if buffer.size < expected or int(message.step) < row_bytes:
        raise ValueError("RGB message buffer is smaller than its dimensions")
    rows = buffer[:expected].reshape(int(message.height), int(message.step))
    pixels = rows[:, :row_bytes].reshape(int(message.height), int(message.width), channels)
    if encoding == "rgb8":
        return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
    if encoding == "mono8":
        return cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
    return pixels.copy()


def depth_message_to_meters(message: Any, depth_scale: float) -> np.ndarray:
    encoding = str(message.encoding).lower()
    if encoding in {"16uc1", "mono16"}:
        dtype = np.dtype(">u2" if bool(message.is_bigendian) else "<u2")
        bytes_per_pixel = 2
        scale = depth_scale
    elif encoding == "32fc1":
        dtype = np.dtype(">f4" if bool(message.is_bigendian) else "<f4")
        bytes_per_pixel = 4
        scale = 1.0
    else:
        raise ValueError(f"Unsupported depth encoding: {message.encoding}")

    row_bytes = int(message.width) * bytes_per_pixel
    raw = np.frombuffer(message.data, dtype=np.uint8)
    expected = int(message.height) * int(message.step)
    if raw.size < expected or int(message.step) < row_bytes:
        raise ValueError("Depth message buffer is smaller than its dimensions")
    rows = raw[:expected].reshape(int(message.height), int(message.step))[:, :row_bytes].copy()
    depth = rows.view(dtype).reshape(int(message.height), int(message.width)).astype(np.float32)
    depth *= float(scale)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def estimate_face_distance(
    depth_meters: np.ndarray | None,
    bbox: Sequence[float],
    rgb_shape: Sequence[int],
    minimum_meters: float = 0.15,
    maximum_meters: float = 10.0,
) -> float | None:
    if depth_meters is None or depth_meters.size == 0:
        return None
    rgb_height, rgb_width = int(rgb_shape[0]), int(rgb_shape[1])
    depth_height, depth_width = depth_meters.shape[:2]
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
    scale_x = depth_width / float(max(1, rgb_width))
    scale_y = depth_height / float(max(1, rgb_height))
    x1, x2 = x1 * scale_x, x2 * scale_x
    y1, y2 = y1 * scale_y, y2 * scale_y

    # Use the central face area to avoid hair, background, and bbox edges.
    roi_x1 = max(0, min(depth_width, int(round(x1 + (x2 - x1) * 0.30))))
    roi_x2 = max(0, min(depth_width, int(round(x2 - (x2 - x1) * 0.30))))
    roi_y1 = max(0, min(depth_height, int(round(y1 + (y2 - y1) * 0.30))))
    roi_y2 = max(0, min(depth_height, int(round(y2 - (y2 - y1) * 0.30))))
    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
        return None
    values = depth_meters[roi_y1:roi_y2, roi_x1:roi_x2].reshape(-1)
    valid = values[(values >= minimum_meters) & (values <= maximum_meters)]
    if valid.size < 5:
        return None
    return float(np.median(valid))


class UdpJsonPublisher:
    def __init__(self, host: str, port: int):
        self.address = (host, int(port)) if port > 0 else None
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if self.address else None

    def publish(self, payload: dict[str, Any]) -> None:
        if not self.socket or not self.address:
            return
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > 60_000:
            raise ValueError("Face-recognition event exceeds the safe UDP datagram size")
        self.socket.sendto(encoded, self.address)

    def close(self) -> None:
        if self.socket:
            self.socket.close()


@dataclass(frozen=True)
class BridgeConfig:
    rgb_topic: str
    depth_topic: str
    result_topic: str
    database: Path
    model_name: str
    model_root: str | None
    providers: tuple[str, ...]
    det_size: int
    threshold: float
    min_margin: float
    max_rate_hz: float
    depth_scale: float
    max_depth_age_seconds: float
    minimum_face_pixels: int
    agent_udp_host: str
    agent_udp_port: int


class FaceRecognitionBridge(Node):  # type: ignore[misc]
    def __init__(self, config: BridgeConfig):
        if ROS_IMPORT_ERROR is not None:
            raise RuntimeError(f"ROS2 Python packages are unavailable: {ROS_IMPORT_ERROR}")
        super().__init__("g1_face_recognition_bridge")
        self.config = config
        with FaceDatabase(config.database) as database:
            samples = database.samples()
        self.gallery = build_gallery(samples)
        if not self.gallery:
            raise ValueError(f"Face gallery is empty: {config.database}")

        available = set(onnxruntime.get_available_providers())
        providers = [provider for provider in config.providers if provider in available]
        if not providers:
            providers = ["CPUExecutionProvider"]
        self.get_logger().info(f"ONNX providers: {providers}; gallery identities: {len(self.gallery)}")
        self.face_app = create_face_app(
            config.model_name,
            config.det_size,
            providers=providers,
            root=config.model_root,
        )

        self.latest_depth: np.ndarray | None = None
        self.latest_depth_stamp_ns = 0
        self.last_inference_monotonic = 0.0
        self.sequence = 0
        self.udp_publisher = UdpJsonPublisher(config.agent_udp_host, config.agent_udp_port)
        self.result_publisher = self.create_publisher(String, config.result_topic, 10)
        self.depth_subscription = self.create_subscription(
            Image,
            config.depth_topic,
            self.on_depth,
            qos_profile_sensor_data,
        )
        self.rgb_subscription = self.create_subscription(
            Image,
            config.rgb_topic,
            self.on_rgb,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"Subscribed RGB={config.rgb_topic}, depth={config.depth_topic}; "
            f"publishing ROS={config.result_topic}, UDP={config.agent_udp_host}:{config.agent_udp_port}"
        )

    def on_depth(self, message: Any) -> None:
        try:
            self.latest_depth = depth_message_to_meters(message, self.config.depth_scale)
            self.latest_depth_stamp_ns = stamp_to_ns(message.header.stamp)
        except ValueError as exc:
            self.get_logger().warning(str(exc))

    def on_rgb(self, message: Any) -> None:
        now = time.monotonic()
        interval = 1.0 / max(0.1, self.config.max_rate_hz)
        if now - self.last_inference_monotonic < interval:
            return
        self.last_inference_monotonic = now
        started = time.perf_counter()
        try:
            image = image_message_to_bgr(message)
        except ValueError as exc:
            self.get_logger().warning(str(exc))
            return

        rgb_stamp_ns = stamp_to_ns(message.header.stamp)
        depth_age_seconds = abs(rgb_stamp_ns - self.latest_depth_stamp_ns) / 1_000_000_000.0
        depth = self.latest_depth if depth_age_seconds <= self.config.max_depth_age_seconds else None
        detected_faces = self.face_app.get(image)
        faces: list[dict[str, Any]] = []
        for index, face in enumerate(detected_faces):
            bbox = face_bbox(face)
            width = float(bbox[2] - bbox[0])
            height = float(bbox[3] - bbox[1])
            if min(width, height) < self.config.minimum_face_pixels:
                continue
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                continue
            match = match_embedding(
                embedding,
                self.gallery,
                threshold=self.config.threshold,
                min_margin=self.config.min_margin,
            )
            distance = estimate_face_distance(depth, bbox, image.shape)
            faces.append(
                {
                    "face_index": index,
                    "person_id": match["person_id"],
                    "status": match["status"],
                    "similarity": round(float(match["similarity"]), 6),
                    "margin": round(float(match["margin"]), 6) if match["margin"] is not None else None,
                    "reason": match["reason"],
                    "det_score": round(float(face.det_score), 6),
                    "bbox": [round(float(value), 2) for value in bbox],
                    "distance_m": round(distance, 3) if distance is not None else None,
                }
            )

        self.sequence += 1
        payload = {
            "schema_version": SCHEMA_VERSION,
            "event": "face_recognition",
            "sequence": self.sequence,
            "timestamp_ns": rgb_stamp_ns,
            "source": {
                "frame_id": str(message.header.frame_id),
                "rgb_topic": self.config.rgb_topic,
                "depth_topic": self.config.depth_topic,
                "depth_age_ms": round(depth_age_seconds * 1000.0, 2) if depth is not None else None,
            },
            "faces": faces,
            "inference_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "model": self.config.model_name,
        }
        ros_message = String()
        ros_message.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self.result_publisher.publish(ros_message)
        try:
            self.udp_publisher.publish(payload)
        except (OSError, ValueError) as exc:
            self.get_logger().warning(f"Agent UDP publish failed: {exc}")

    def destroy_node(self) -> bool:
        self.udp_publisher.close()
        return super().destroy_node()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--result-topic", default="/ai/face_recognition/results")
    parser.add_argument("--db", type=Path, default=Path("/data/team_faces.sqlite3"))
    parser.add_argument("--model", default="buffalo_l")
    parser.add_argument("--model-root", default="/models")
    parser.add_argument("--providers", default="CUDAExecutionProvider,CPUExecutionProvider")
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--min-margin", type=float, default=0.05)
    parser.add_argument("--max-rate-hz", type=float, default=5.0)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--max-depth-age", type=float, default=0.25)
    parser.add_argument("--minimum-face-pixels", type=int, default=48)
    parser.add_argument("--agent-udp-host", default="127.0.0.1")
    parser.add_argument("--agent-udp-port", type=int, default=17171)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if ROS_IMPORT_ERROR is not None:
        print(f"ROS2 Python packages are required: {ROS_IMPORT_ERROR}")
        return 2
    parser = build_parser()
    args, ros_args = parser.parse_known_args(argv)
    providers = tuple(item.strip() for item in args.providers.split(",") if item.strip())
    config = BridgeConfig(
        rgb_topic=args.rgb_topic,
        depth_topic=args.depth_topic,
        result_topic=args.result_topic,
        database=args.db.resolve(),
        model_name=args.model,
        model_root=args.model_root,
        providers=providers,
        det_size=args.det_size,
        threshold=args.threshold,
        min_margin=args.min_margin,
        max_rate_hz=args.max_rate_hz,
        depth_scale=args.depth_scale,
        max_depth_age_seconds=args.max_depth_age,
        minimum_face_pixels=args.minimum_face_pixels,
        agent_udp_host=args.agent_udp_host,
        agent_udp_port=args.agent_udp_port,
    )
    rclpy.init(args=ros_args)
    node: FaceRecognitionBridge | None = None
    try:
        node = FaceRecognitionBridge(config)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
