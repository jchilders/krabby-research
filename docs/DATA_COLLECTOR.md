# HAL data collector (rosbag2 / mcap)

The collector exists so you can **keep a time-aligned trace of what the robot (or sim) actually sensed and reported** while policies run—useful for debugging behaviors, comparing runs, building datasets, and replaying or analyzing data later with normal ROS 2 bag tooling. It turns the same **`HardwareObservations`** stream the policy already consumes into **standard rosbag2 v9 / mcap** files without running a separate recorder service or a full ROS 2 distro in the container (**[rosbags](https://pypi.org/project/rosbags/)** writes the bags).

That is done with a **second `HalClient`** in the **same process** as the primary client (Jetson `hal.server.jetson.main` or Isaac `hal.server.isaac.main`), on the **same in-proc ZMQ endpoints** and shared **`zmq.Context`**, so the recording matches the observation contract the stack already publishes.

## Run (Jetson or Isaac Sim container)

1. Build **locomotion** (`images/locomotion/Dockerfile`) or **Isaac Sim** (`images/isaacsim/Dockerfile`). Both images include `data_collection/` and pin **`rosbags`** (and **`PyYAML`** for optional `load_config` / tests) in their `requirements.txt`.
2. Adjust recording defaults in **`data_collection/collector_settings.py`** (same pattern as camera rows in **`hal/server/jetson/sensor_backend_jetson.py`**).
3. Pass **`--data-collector-output-dir <container_path>`** to enable recording.
4. Mount a host directory to that same container path so bags persist after container exit.

### Host mount requirement

Recordings are written to the container filesystem unless you mount a host path. Use a bind mount so rosbags persist after container exit.

Create a host directory once:

```bash
mkdir -p /path/to/krabby_bags
```

### Jetson (locomotion image)

Collector output path in container (`/workspace/bags`):

```bash
docker run --rm --runtime=nvidia \
  -v /path/to/checkpoints:/workspace/checkpoints \
  -v /path/to/krabby_bags:/workspace/bags \
  -v /dev:/dev \
  --privileged \
  krabby-locomotion:latest \
  --checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt \
  --data-collector-output-dir /workspace/bags
```

### Isaac Sim image (`krabby-isaacsim`)

Collector output path in container (`/workspace/bags`):

```bash
docker run --rm --gpus all \
  -e ACCEPT_EULA=Y \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /path/to/checkpoints:/workspace/checkpoints \
  -v /path/to/krabby_bags:/workspace/bags \
  krabby-isaacsim:latest \
  --task 'Your-Task-v0' \
  --checkpoint /workspace/checkpoints/your.ckpt \
  --data-collector-output-dir /workspace/bags
```

If you launch from an interactive shell inside a running container (instead of `docker run`), pass `--data-collector-output-dir`; the chosen output path must be on a mounted volume if you want host persistence.

## Configuration (Python)

Recording parameters live in **`data_collection/collector_settings.py`**. The entrypoint calls **`build_data_collector_config(observation_endpoint=..., command_endpoint=..., output_dir=...)`** so collector endpoints match the primary HAL client endpoints. On Jetson (`hal.server.jetson.main`), these are fixed in-process endpoints; on Isaac (`hal.server.isaac.main`), they may come from CLI bind args. The dataclass shape is **`DataCollectorConfig`** in **`data_collection/config.py`**; **`DataCollectorConfig.from_dict`** / **`load_config`** remain for tests or ad hoc YAML if you build your own loader.

| Area | Where it is set |
|------|-----------------|
| HAL transport | From the server's configured endpoints via **`build_data_collector_config`** (must match primary `HalClientConfig`; Jetson uses fixed inproc endpoints). |
| Output | **`DEFAULT_OUTPUT_DIR`**, **`MAX_DISK_USAGE_FRACTION`**, **`ROTATION_MAX_BYTES`**, **`ROTATION_MAX_MINUTES`** in **`collector_settings.py`**; override directory with **`--data-collector-output-dir`**. |
| Rates | **`RECORDING_RATES`** — wall-clock caps; actual rate is still bounded by HAL publish rate and **latest-only** `poll()` semantics. |
| Topics | **`TOPIC_ENABLE`** — booleans per ROS topic (see table below). |
| Catalog | **`CATALOG_TOPIC_MAP`** — Jetson **`rgbd_by_catalog_id`** keys (e.g. `side_rgbd`). |
| Commands | **`joints_command_source`** is only **`previous_action`** in code — `/joints/command` is filled from **`HardwareObservations.previous_action`**, not from a separate tap on `put_joint_command`. |
| Joints | **`JOINT_NAMES`** — if empty or length mismatch, serialization uses `joint_0`, `joint_1`, … |
| Polling | **`POLLING_TIMEOUT_MS`** — ZMQ receive timeout per collector poll. |

## Topic map (HAL → ROS)

| HAL source | ROS topic | Message type |
|------------|-----------|--------------|
| `camera_rgb` | `/camera/front/rgb` | `sensor_msgs/Image` (`rgb8`) |
| `camera_depth` | `/camera/front/depth` | `sensor_msgs/Image` (`32FC1`, meters) |
| `rgbd_by_catalog_id[id].rgb` | `/camera/side_left/rgb` | `sensor_msgs/Image` — when `catalog_map.side_left_rgb_catalog_id` matches, or legacy `side_camera_rgb` if no catalog row wrote that topic yet |
| `rgbd_by_catalog_id[id].rgb` | `/camera/side_right/rgb` | when `side_right_rgb_catalog_id` matches |
| `rgbd_by_catalog_id[id].depth` | `/camera/side_rgbd/depth` | `sensor_msgs/Image` (`32FC1`) when `side_rgbd_depth_catalog_id` matches |
| `rgbd_by_catalog_id[radar].rgb` (mono) | `/radar/edge` | `sensor_msgs/Image` (`mono8`) when `radar_edge_catalog_id` is set |
| `joint_positions`, `joint_velocities` | `/joints/state` | `sensor_msgs/JointState` |
| `previous_action` | `/joints/command` | `sensor_msgs/JointState` (positions only; see above) |
| `base_quat_w`, `base_ang_vel_b` | `/imu` | `sensor_msgs/Imu` — orientation + angular velocity; linear acceleration is zero (not estimated here). |

## Playback

- **ROS 2:** `ros2 bag info` / `ros2 bag play` on the bag directory.
- **This repo:** `python scripts/playback_krabby_bag.py <bag_dir> --topic /camera/front/rgb --max 5` (optional `--display` if OpenCV is installed).

## Architecture

The collector is not a separate service. When **`--data-collector-output-dir`** is set, the HAL server entrypoint (`hal.server.jetson.main` or `hal.server.isaac.main`) constructs a **second `HalClient`** in the **same process** as the policy client, sharing the **`zmq.Context`** and using the **same** observation and command endpoint strings as the primary `HalClientConfig`. Both subscribers receive the same published `HardwareObservations`; the collector only serializes and writes them. That keeps recordings aligned with what the stack already exposes, avoids an extra HAL bridge or network hop, and leaves HAL’s contract unchanged—recording is an optional side path in the existing process.

Serialization and bags use **[rosbags](https://pypi.org/project/rosbags/)** (rosbag2 layout, no full ROS 2 distro required). If `rosbags` cannot be imported, the writer is skipped and the rest of the server still runs.

## Recording lifecycle

Under `output_dir`, each **segment** is a standard **rosbag2 v9** directory with **mcap** storage, named `krabby_<seq>_<YYYYMMDD_HHMMSS>`, containing `metadata.yaml` and mcap data—compatible with `ros2 bag info` / `ros2 bag play` and the repo playback helper in the section above.

**Rotation:** when the current segment’s on-disk size reaches `rotation_max_bytes` **or** its age reaches `rotation_max_minutes`, the writer closes that directory and opens a new one (sequence increments).

**Disk cap:** `max_disk_usage_fraction` applies to the **filesystem** that backs `output_dir` (`total` capacity from the OS). The implementation sums sizes of all bag directories under `output_dir` (those with `metadata.yaml`) and deletes **oldest** bags first until usage is under the limit. Mount `output_dir` on a volume sized for the retention you want.

**Effective sample rate:** `rates.images_hz` and `rates.joints_imu_hz` cap how often the collector samples; together with HAL’s **latest-only** subscriber behavior and the publisher’s own rate, the bag may contain fewer messages than a naive “Hz × duration” estimate.

## Configuration semantics

- **`topics.*`:** Only enabled topics are registered on the bag writer, so disabling a stream removes that topic from the segment entirely.
- **`catalog_map`:** Values are **catalog ids** matching keys in `HardwareObservations.rgbd_by_catalog_id` on Jetson (authoritative list: `JETSON_SENSOR_CATALOG` in `hal/server/jetson/sensor_backend_jetson.py`; overview in [SENSOR_INTERFACE.md](SENSOR_INTERFACE.md)). Use `null` to turn off an optional stream (e.g. no side-right RGB until that camera exists). Side-left RGB can still fall back from legacy `side_camera_rgb` when catalog routing has not produced `/camera/side_left/rgb` yet (see `data_collection/serialization.py`).
- **HAL endpoints:** Wrong or mismatched strings mean no observations or the wrong socket; **`build_data_collector_config`** must receive the same bind strings as the primary client.
- **Source of truth:** Edit **`data_collection/collector_settings.py`** for defaults (mirrors how camera rows live in Python on the HAL side).
