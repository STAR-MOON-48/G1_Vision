import importlib.util
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


EXAMPLE_DIR = Path(__file__).parent
sys.path.insert(0, str(EXAMPLE_DIR))
SPEC = importlib.util.spec_from_file_location("g1_face_bridge", EXAMPLE_DIR / "g1_face_bridge.py")
assert SPEC and SPEC.loader
BRIDGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BRIDGE
SPEC.loader.exec_module(BRIDGE)


def fake_image(array, encoding, stamp=(1, 2), frame_id="camera"):
    return SimpleNamespace(
        width=array.shape[1],
        height=array.shape[0],
        encoding=encoding,
        is_bigendian=False,
        step=array.strides[0],
        data=array.tobytes(),
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=stamp[0], nanosec=stamp[1]),
            frame_id=frame_id,
        ),
    )


class G1FaceBridgeTests(unittest.TestCase):
    def test_rgb_message_conversion(self):
        rgb = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
        bgr = BRIDGE.image_message_to_bgr(fake_image(rgb, "rgb8"))
        np.testing.assert_array_equal(bgr[0, 0], np.array([0, 0, 255]))
        np.testing.assert_array_equal(bgr[0, 1], np.array([0, 255, 0]))

    def test_depth_conversion_and_face_median(self):
        depth_mm = np.full((10, 10), 1500, dtype=np.uint16)
        depth_mm[0, 0] = 0
        depth = BRIDGE.depth_message_to_meters(fake_image(depth_mm, "16UC1"), 0.001)
        distance = BRIDGE.estimate_face_distance(depth, [0, 0, 10, 10], (10, 10, 3))
        self.assertAlmostEqual(distance, 1.5, places=3)

    def test_udp_json_publisher(self):
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(1.0)
        publisher = BRIDGE.UdpJsonPublisher("127.0.0.1", receiver.getsockname()[1])
        try:
            publisher.publish({"event": "face_recognition", "faces": []})
            payload, _address = receiver.recvfrom(4096)
            self.assertIn(b"face_recognition", payload)
        finally:
            publisher.close()
            receiver.close()

    def test_stamp_to_ns(self):
        self.assertEqual(BRIDGE.stamp_to_ns(SimpleNamespace(sec=2, nanosec=3)), 2_000_000_003)


if __name__ == "__main__":
    unittest.main()
