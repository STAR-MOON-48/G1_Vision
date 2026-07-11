# Team face recognition demo

This demo implements the recognition core described in `docs/NEW_FACE.MD`:

1. detect and crop the primary face in each enrollment photo;
2. extract normalized InsightFace embeddings;
3. store stable person IDs and embeddings in SQLite;
4. recognize faces with a cosine threshold and an optional Top-1/Top-2 margin;
5. return `UNKNOWN` when the match is not strong enough.

The model is not retrained. Generated crops and the SQLite database stay under
`.face_demo/`, which is intentionally excluded from Git.

## Windows quick start

Create a virtual environment and install this checkout:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .\python-package
```

Preprocess all `FACE_TEAM_*` identity folders under `face_datasets`. Identities
with fewer than 15 source photos receive mild deterministic augmentations;
`test_data` is always excluded from enrollment:

```powershell
.\.venv\Scripts\python.exe .\examples\team_face_recognition\team_face_demo.py preprocess
```

Extract embeddings and create the SQLite gallery:

```powershell
.\.venv\Scripts\python.exe .\examples\team_face_recognition\team_face_demo.py enroll --replace
```

Validate each original photo while excluding that photo and all augmentations
derived from it from the gallery:

```powershell
.\.venv\Scripts\python.exe .\examples\team_face_recognition\team_face_demo.py validate
```

Recognize a separate query image and save an annotated result:

```powershell
.\.venv\Scripts\python.exe .\examples\team_face_recognition\team_face_demo.py recognize `
  --image C:\path\to\query.jpg `
  --output .\.face_demo\recognized.jpg `
  --json-output .\.face_demo\recognized.json
```

Every `FACE_TEAM_*` subdirectory of `face_datasets` is treated as a stable
person ID, such as `FACE_TEAM_001`. Do not use an enrollment image as the only
recognition test: add independent photos from the intended G1 camera distance
and add at least one non-team person before selecting the production threshold.

The initial threshold is `0.40`, matching this checkout's GUI default. It is a
starting value, not a calibrated security threshold.

## G1 to DGX bridge

真机日常启动、Agent 订阅、验收和故障排查请参阅
[`docs/G1_FACE_RECOGNITION_USER_GUIDE.md`](../../docs/G1_FACE_RECOGNITION_USER_GUIDE.md)。

`g1_face_bridge.py` subscribes ROS2 RGB and aligned-depth images, runs the same
SQLite gallery matcher, publishes internal JSON on
`/ai/face_recognition/internal`, relays native `g1_hri.msg.FaceRecognition` on
`rt/g1/hri/vision/face_recognition`, and
also sends the JSON event to a local Agent over UDP `127.0.0.1:17171`.

The reproducible DGX Spark container deployment is under `deploy/`. Camera
transport is isolated on ROS domain 10 and uses Fast DDS on both ORIN and DGX;
native Unitree Agent events remain on domain 0 and interface `enP7s7`. Topic
names and output addresses are configured in `deploy/.env`.

Agent-facing results use native Unitree SDK2 DDS rather than a ROS2 message:

- Topic: `rt/g1/hri/vision/face_recognition`
- Type: `g1_hri.msg.FaceRecognition`
- Relay: user service `g1-face-dds-relay.service`
- Agent example: `agent_face_subscriber.py enP7s7 --domain 0`

The ROS2 `std_msgs/String` topic `/ai/face_recognition/internal` is reserved for
container diagnostics so it cannot conflict with the native DDS topic type.
