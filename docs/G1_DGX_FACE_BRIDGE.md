# G1 - DGX Spark face-recognition bridge

For the validated startup, shutdown, Agent subscription, acceptance test and
troubleshooting procedures, see
[`G1_FACE_RECOGNITION_USER_GUIDE.md`](G1_FACE_RECOGNITION_USER_GUIDE.md).

## Confirmed environment

- DGX Spark: ARM64, Ubuntu 24.04, NVIDIA GB10, CUDA 13 driver.
- G1-facing interface: `enP7s7`, address `192.168.123.100/24`.
- Internet/Windows-facing interface: `wlP9s9`, address `172.16.21.132/22`.
- G1/Orin endpoint `192.168.123.164` is reachable with sub-millisecond latency.
- Unitree control/HRI RTPS traffic is present on DDS domain 0; camera transport
  is isolated on domain 10.

## Data flow

```text
G1 D435i RGB + aligned depth
  -> ROS2/DDS camera domain 10 over 192.168.123.0/24
  -> g1_face_recognition_bridge on DGX Spark
  -> InsightFace detector and two-person SQLite gallery
  -> internal ROS2 JSON topic /ai/face_recognition/internal
  -> localhost UDP relay
  -> native DDS g1_hri.msg.FaceRecognition
     topic rt/g1/hri/vision/face_recognition
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
    "rgb_topic": "/camera/color/image_raw",
    "depth_topic": "/camera/aligned_depth_to_color/image_raw",
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

The Docker container uses host networking. Camera transport is isolated on
domain 10 and uses Fast DDS on both the Foxy ORIN bridge and the Jazzy DGX
subscriber. Agent-facing Unitree SDK2 DDS remains isolated on domain 0 and is
bound to `enP7s7` by the native relay.

## Confirmed live topics

The live D435i integration was validated on 2026-07-11 with these topics:

- RGB: `/camera/color/image_raw`
- Aligned depth: `/camera/aligned_depth_to_color/image_raw`
- Agent result: `rt/g1/hri/vision/face_recognition`

The end-to-end test recognized `FACE_TEAM_001` and returned a valid depth
distance through the native DDS Agent subscriber.
