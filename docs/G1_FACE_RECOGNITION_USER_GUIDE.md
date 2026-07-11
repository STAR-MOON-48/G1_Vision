# 宇树 G1 人脸识别系统使用说明

本文说明如何使用已经部署在 G1 ORIN NX 和 NVIDIA DGX Spark 上的人脸识别系统。内容以 2026-07-11 的真机联调结果为准。

## 1. 系统用途

系统从 G1 头部 Intel RealSense D435i 获取 RGB 和对齐深度图，在 DGX Spark 上运行 InsightFace，识别 SQLite 人脸库中的团队成员，并把结果通过 Unitree SDK2 原生 DDS 发布给本地 Agent。

当前人脸库包含：

- `FACE_TEAM_001`
- `FACE_TEAM_002`

系统不会重新训练 InsightFace 模型。录入新成员时只需提取人脸特征并更新 SQLite 人脸库。

## 2. 已验证的设备与网络

| 设备或接口 | 地址/配置 | 用途 |
|---|---|---|
| G1 ORIN NX | `192.168.123.164` | 采集 D435i RGB/Depth，并转发到 DDS |
| DGX 的 G1 网口 | `enP7s7` / `192.168.123.100` | 接收相机流，发布 Agent DDS 结果 |
| DGX 管理网口 | `172.16.21.132` | Windows 开发机 SSH 登录 DGX |
| 相机 DDS | Domain `10`，Fast DDS | 传输高带宽 RGB/Depth 图像 |
| Agent DDS | Domain `0`，Unitree SDK2 | 发布轻量人脸识别事件 |

从 Windows 登录时，先进入 DGX，再由 DGX 登录 G1：

```bash
ssh dgx@172.16.21.132
ssh unitree@192.168.123.164
```

不要把登录密码、SSH 私钥、人脸照片、embedding 或 SQLite 人脸库提交到 Git。

## 3. 数据链路

```text
D435i RGB + aligned depth
  -> ORIN ROS1 Noetic realsense2_camera
  -> ORIN ROS1-to-ROS2 bridge
  -> Fast DDS / Domain 10
  -> DGX g1_face_recognition_bridge
  -> InsightFace + SQLite gallery
  -> localhost UDP 127.0.0.1:17171
  -> g1-face-dds-relay.service
  -> Unitree SDK2 DDS / Domain 0
  -> rt/g1/hri/vision/face_recognition
  -> Local Agent
```

相机图像不会发送给 Agent。Agent 只接收人员 ID、匹配分数、人脸框和距离等结构化结果。

## 4. Topic 和服务

### 4.1 相机 Topic（Domain 10）

| Topic | 类型 | 内容 |
|---|---|---|
| `/camera/color/image_raw` | `sensor_msgs/msg/Image` | 640×480 RGB |
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/msg/Image` | 与 RGB 对齐的 640×480 深度图 |
| `/ai/face_recognition/internal` | `std_msgs/msg/String` | 容器内部诊断 JSON |

相机当前按 640×480、15 FPS 请求。受现有 USB 线和 USB 2.0 链路影响，真机测得对齐深度约 9–10 FPS。

### 4.2 Agent Topic（Domain 0）

```text
Topic: rt/g1/hri/vision/face_recognition
Type:  g1_hri.msg.FaceRecognition
```

该 Topic 与以下语音 Topic 相互独立：

```text
rt/g1/hri/speech/final
rt/g1/hri/playback/state
```

语音 Topic 出现类型哈希警告，不代表视觉链路失败。

### 4.3 常驻服务

ORIN 用户服务：

```text
g1-realsense-ros1.service
g1-realsense-dds-bridge.service
```

DGX 服务：

```text
Docker: deploy-g1-face-bridge-1
systemd user: g1-face-dds-relay.service
```

## 5. 日常启动流程

建议按“G1 相机 → ORIN bridge → DGX 算法 → Agent”的顺序检查。

### 5.1 检查 G1 和 D435i

在 DGX 上执行：

```bash
ping -c 3 192.168.123.164
ssh unitree@192.168.123.164
```

进入 ORIN 后检查相机：

```bash
lsusb | grep -i RealSense
```

正常输出应包含：

```text
8086:0b3a Intel Corp. Intel(R) RealSense(TM) Depth Camera 435i
```

启动并检查两个用户服务：

```bash
systemctl --user start g1-realsense-ros1.service
systemctl --user start g1-realsense-dds-bridge.service

systemctl --user status g1-realsense-ros1.service --no-pager
systemctl --user status g1-realsense-dds-bridge.service --no-pager
```

两项都应显示 `active (running)`。

检查相机在 ORIN 本机是否持续出图：

```bash
source /opt/ros/noetic/setup.bash
rostopic hz /camera/color/image_raw
rostopic hz /camera/aligned_depth_to_color/image_raw
```

使用 `Ctrl+C` 结束频率检查。

### 5.2 启动 DGX 人脸识别容器

退出 ORIN、回到 DGX：

```bash
cd ~/insightface-g1/examples/team_face_recognition/deploy
docker compose up -d
docker compose ps
docker compose logs --tail 50 g1-face-bridge
```

日志中应出现：

```text
gallery identities: 2
Subscribed RGB=/camera/color/image_raw, depth=/camera/aligned_depth_to_color/image_raw
```

容器必须使用：

```text
ROS_DOMAIN_ID=10
RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

可用下面的命令确认：

```bash
docker inspect deploy-g1-face-bridge-1 \
  --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep -E 'ROS_DOMAIN_ID|RMW_IMPLEMENTATION'
```

### 5.3 启动 Agent DDS relay

在 DGX 上执行：

```bash
systemctl --user start g1-face-dds-relay.service
systemctl --user status g1-face-dds-relay.service --no-pager
```

查看日志：

```bash
journalctl --user -u g1-face-dds-relay.service -n 50 --no-pager
```

正常日志包含：

```text
DDS FaceRecognition publisher: rt/g1/hri/vision/face_recognition
Face UDP->DDS relay listening on 127.0.0.1:17171
```

## 6. Agent 订阅方法

在 DGX 上打开一个新终端：

```bash
cd ~/insightface-g1/examples/team_face_recognition

/home/dgx/moonbot/speech_service/.venv/bin/python -u \
  agent_face_subscriber.py enP7s7 --domain 0
```

程序会持续等待 DDS 消息，不会自动退出。没有输出通常表示暂时没有收到事件；使用 `Ctrl+C` 结束。

识别到成员时示例：

```text
Agent perception <- [...] frame=camera_color_optical_frame faces=[{'person_id': 'FACE_TEAM_001', 'status': 'KNOWN', 'similarity': 0.458671, 'distance_m': 0.598, ...}]
```

常见结果：

- `faces=[]`：相机流和算法正在工作，但当前画面没有检测到完整人脸。
- `status='KNOWN'`：检测到人脸并匹配到已录入成员。
- `status='UNKNOWN'`：检测到人脸，但相似度或 Top-1/Top-2 margin 未达到阈值。
- `distance_m=null`：没有取得足够新的、有效的对齐深度数据。

## 7. 人员站位要求

当前 G1 头部 D435i 视角偏下。真机测试中，正常站立时画面只拍到身体，人脸位于画面上方之外，因此持续返回 `faces=[]`。

测试时建议：

1. 人脸正对 G1 头部相机；
2. 距离保持约 0.7–1.5 米；
3. 蹲下或弯腰，使完整人脸进入画面中央；
4. 避免强逆光、口罩和快速转头；
5. 保持至少 2–3 秒，观察连续多帧结果。

如果 `bbox` 的上边界为负数，例如 `y=-68`，说明人脸仍有一部分在画面上方之外，应继续降低站位或调整相机角度。

## 8. 无 G1 的 smoke test

`smoke_test.py` 用合成图像验证 DGX 容器、InsightFace 初始化和 JSON/UDP 输出，不需要 G1 开机，也不代表真实相机链路已经打通。

```bash
docker exec deploy-g1-face-bridge-1 bash -lc \
  "source /opt/ros/jazzy/setup.bash && \
   python3 /app/team_face_recognition/deploy/smoke_test.py"
```

成功标志：

```text
SMOKE_OK
```

真机验收仍必须打开 G1 和 D435i，并使用 Agent DDS subscriber 检查真实识别结果。

## 9. 真机验收标准

一次完整验收应同时满足：

1. ORIN 两个服务均为 `active (running)`；
2. bridge 连续运行至少一分钟且 `NRestarts=0`；
3. DGX 容器持续运行；
4. DDS relay 为 `active`；
5. Agent 能连续收到 `frame=camera_color_optical_frame` 事件；
6. 已录入成员在合理站位下返回 `status=KNOWN`；
7. 有效 Depth 时返回非空 `distance_m`。

检查 bridge 重启次数：

```bash
ssh unitree@192.168.123.164 \
  "systemctl --user show g1-realsense-dds-bridge.service \
   -p NRestarts -p ActiveEnterTimestamp"
```

2026-07-11 真机验收结果：

```text
person_id:  FACE_TEAM_001
status:     KNOWN
similarity: 0.458671
distance_m: 0.598
frame_id:   camera_color_optical_frame
```

## 10. 停止与重启

### 10.1 重启 DGX 算法

```bash
cd ~/insightface-g1/examples/team_face_recognition/deploy
docker compose restart g1-face-bridge
systemctl --user restart g1-face-dds-relay.service
```

### 10.2 重启 ORIN 相机链路

在 ORIN 上执行：

```bash
systemctl --user restart g1-realsense-ros1.service
systemctl --user restart g1-realsense-dds-bridge.service
```

### 10.3 完全停止

DGX：

```bash
cd ~/insightface-g1/examples/team_face_recognition/deploy
docker compose stop
systemctl --user stop g1-face-dds-relay.service
```

ORIN：

```bash
systemctl --user stop g1-realsense-dds-bridge.service
systemctl --user stop g1-realsense-ros1.service
```

## 11. 常见故障排查

### 11.1 Agent 一直没有输出

按顺序检查：

```bash
systemctl --user is-active g1-face-dds-relay.service
docker ps --filter name=deploy-g1-face-bridge-1
ping -c 3 192.168.123.164
```

然后检查 ORIN 的相机和 bridge 服务。

### 11.2 持续 `faces=[]`

这通常不是通信故障。只要事件中的 `frame` 是 `camera_color_optical_frame`，就说明 RGB 已经到达算法。

优先检查：

- 人脸是否完整进入画面；
- G1 相机是否朝下；
- 人脸是否太远或太小；
- 是否有遮挡、逆光或运动模糊。

### 11.3 检测到人脸但返回 `UNKNOWN`

先改善站位和光线，不要立即降低阈值。默认阈值为 `0.40`，默认最小 margin 为 `0.05`。生产部署前应加入团队外人员数据进行阈值标定。

### 11.4 RGB 正常但 `distance_m` 为空

在 ORIN 检查：

```bash
source /opt/ros/noetic/setup.bash
rostopic hz /camera/aligned_depth_to_color/image_raw
```

如果 ORIN 本机有 Depth，但 DGX 长时间没有距离，检查 `g1-realsense-dds-bridge.service` 是否稳定、有无重启。

### 11.5 bridge 出现 `std::bad_alloc` 或不断重启

确认相机域两端都使用 Fast DDS：

```text
RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ROS_DOMAIN_ID=10
```

ORIN bridge 和 DGX 人脸节点必须关闭 ROS2 `/rosout` 发布，避免 Foxy 与 Jazzy 自动建立不兼容的反向日志桥。项目中的脚本和节点已经包含该修复：

```text
deploy/orin/start_ros1_bridge.sh: --disable-rosout-logs
g1_face_bridge.py: enable_rosout=False
```

不要改回 CycloneDDS，也不要删除上述 `/rosout` 隔离参数。

### 11.6 ROS2 显示相机 Publisher count 为 0

G1 未开机时这是正常现象。G1 已开机时检查：

1. DGX 是否能 ping 通 `192.168.123.164`；
2. D435i 是否出现在 `lsusb`；
3. ORIN RealSense 服务是否运行；
4. ORIN bridge 是否运行；
5. 两端是否都使用 Domain 10。

## 12. 消息字段说明

| 字段 | 含义 |
|---|---|
| `event_id` | 全局唯一事件 ID，供 Agent 去重 |
| `session_id` | 当前视觉服务运行会话 |
| `sequence` | 会话内递增序号 |
| `created_unix_ns` | 事件创建时间 |
| `frame_id` | 相机坐标帧，正常为 `camera_color_optical_frame` |
| `faces` | 当前帧检测到的人脸列表 |
| `person_id` | 已录入成员 ID；未知人员为 `null` |
| `status` | `KNOWN` 或 `UNKNOWN` |
| `similarity` | 查询人脸与最佳模板的余弦相似度 |
| `margin` | Top-1 与其他身份候选之间的区分度 |
| `reason` | `matched`、`below_threshold` 等判断原因 |
| `det_score` | InsightFace 人脸检测置信度 |
| `bbox` | RGB 图像中的人脸框 `[x1,y1,x2,y2]` |
| `distance_m` | 对齐深度图估算的人脸距离，单位为米 |
| `inference_ms` | 当前帧推理耗时 |

## 13. 文件位置

DGX 项目：

```text
~/insightface-g1/examples/team_face_recognition/
```

DGX 运行数据：

```text
deploy/runtime/models/buffalo_l/
deploy/runtime/data/team_faces.sqlite3
```

ORIN 运行脚本：

```text
/home/unitree/g1_camera_bridge/start_realsense_ros1.sh
/home/unitree/g1_camera_bridge/start_ros1_bridge.sh
```

ORIN ROS1 bridge overlay：

```text
/home/unitree/ros1_bridge_overlay/
```

## 14. 硬件注意事项

当前 D435i Type-C 线存在明显破损，并且相机重新插入后可能从 USB 3.0 降为 USB 2.0。裸露屏蔽层或导线可能造成短路、掉线和图像流不稳定。

- 不要在机器人运动或带载时拉动线缆；
- 调试期间保持 G1 头部和线缆静止；
- 插拔前优先关闭相关服务并断开相机供电；
- 如果再次出现掉线，先检查 `lsusb` 和系统日志，不要先修改识别算法。

