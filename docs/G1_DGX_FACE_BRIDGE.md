# G1 - DGX Spark face-recognition bridge

## Confirmed environment

- DGX Spark: ARM64, Ubuntu 24.04, NVIDIA GB10, CUDA 13 driver.
- G1-facing interface: `enP7s7`, address `192.168.123.100/24`.
- Internet/Windows-facing interface: `wlP9s9`, address `172.16.21.132/22`.
- G1/Orin endpoint `192.168.123.164` is reachable with sub-millisecond latency.
- RTPS discovery traffic is present on DDS domain 0.

## Data flow

```text
G1 D435i RGB + aligned depth
  -> ROS2/DDS domain 0 over 192.168.123.0/24
  -> g1_face_recognition_bridge on DGX Spark
  -> InsightFace detector and two-person SQLite gallery
  -> ROS2 JSON topic /ai/face_recognition/results
  -> local UDP JSON 127.0.0.1:17171
  -> local Agent
```

The bridge never sends images or embeddings to the Agent. Each event contains
only face boxes, stable person IDs, match scores, rejection status, and an
optional median face distance derived from aligned depth.

## Result schema

```json
{
  "schema_version": 1,
  "event": "face_recognition",
  "sequence": 42,
  "timestamp_ns": 1783742400000000000,
  "source": {
    "frame_id": "camera_color_optical_frame",
    "rgb_topic": "/camera/camera/color/image_raw",
    "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
    "depth_age_ms": 18.2
  },
  "faces": [
    {
      "face_index": 0,
      "person_id": "FACE_TEAM_001",
      "status": "KNOWN",
      "similarity": 0.896117,
      "margin": 0.821183,
      "reason": "matched",
      "det_score": 0.781573,
      "bbox": [694.07, 999.25, 2153.16, 3045.44],
      "distance_m": 1.42
    }
  ],
  "inference_ms": 73.5,
  "model": "buffalo_l"
}
```

## Deployment layout on DGX

```text
~/insightface-g1/
  python-package/
  examples/team_face_recognition/
    deploy/
      compose.yaml
      cyclonedds.xml
      runtime/
        models/buffalo_l/{det_10g.onnx,w600k_r50.onnx}
        data/team_faces.sqlite3
```

The Docker container uses host networking so DDS multicast remains on the
physical G1 LAN. `CYCLONEDDS_URI` pins discovery and traffic to `enP7s7`; the
default route stays on `wlP9s9` for internet access.

## Remaining integration check

The exact D435i RGB and aligned-depth ROS topic names must be discovered on the
live network before starting the final service. The defaults match the standard
`realsense2_camera` namespace, but the G1 publisher may use a custom namespace.
