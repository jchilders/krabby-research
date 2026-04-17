# Jetson Deployment Guide

This guide explains how to build and deploy the parkour policy runtime on Jetson Orin hardware.

## System Requirements

- **Device**: NVIDIA Jetson Orin
- **OS**: Ubuntu 22.04.5 LTS (Jammy Jellyfish)
- **Kernel**: 5.15.148-tegra (NVIDIA Tegra)
- **Architecture**: aarch64 (ARM64)
- **JetPack**: 6.1/6.2 (L4T 36.4) or later
- **ZED 2i Camera**: Optional RGB-D (ZED SDK 5.1.1+ for L4T 36.4)
- **MaixSense-A075V**: Optional RGB-D over USB RNDIS + HTTP (no ZED SDK); see [MaixSense-A075V (optional host bring-up)](#maixsense-a075v-optional-host-bring-up) below
- **Model checkpoint file**: `.pt` format (e.g., `unitree_go2_parkour_teacher.pt`)

### Seeed reComputer Jetson Robotics J401 (reference carrier)

If you deploy on **Seeed Studio**’s [reComputer Jetson Robotics J401](https://wiki.seeedstudio.com/recomputer_jetson_robotics_J401_getting_started/), use their documentation alongside this guide:

- **Getting started:** [reComputer Jetson Robotics J401 — Seeed Wiki](https://wiki.seeedstudio.com/recomputer_jetson_robotics_J401_getting_started/)
- **Hardware reference:** [reComputer Jetson Robotics J401 datasheet (PDF)](https://files.seeedstudio.com/products/NVIDIA-Jetson/reComputer_robotics_J401_datasheet.pdf)

## Prerequisites: System Setup

Before building and deploying, ensure the Jetson system is properly configured.

### 1. SSH Access

You need SSH access to the Jetson from your development machine (e.g., to copy images, run commands, or deploy).

**On the Jetson**: Ensure the SSH server is installed and running:

```bash
# Install OpenSSH server if not present
sudo apt-get update
sudo apt-get install -y openssh-server
sudo systemctl enable ssh
sudo systemctl start ssh
```

**On your local machine**: Configure SSH so you can connect by hostname or alias. Edit or create `~/.ssh/config` (on Windows: `C:\Users\<YourUsername>\.ssh\config`):

```
Host jetson
    HostName <hostname-or-ip>
    User <username>
    # Optional: key-based authentication
    # IdentityFile ~/.ssh/id_rsa
```

Replace `<hostname-or-ip>` with the Jetson’s hostname (if it resolves on your network) or its IP address. Replace `<username>` with your Jetson user account.

**Test the connection:**

```bash
ssh jetson
# Or, if not using the config alias:
ssh <username>@<hostname-or-ip>
```

**Troubleshooting:**

- If the connection fails, ensure the Jetson is on the same network and reachable (e.g., `ping <hostname-or-ip>`).
- Ensure the SSH server is running on the Jetson: `sudo systemctl status ssh`.
- Check that the firewall allows SSH (port 22).

### 2. Docker Installation

Docker is required for running the locomotion container.

```bash
# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add user to docker group (to run docker without sudo)
sudo usermod -aG docker $USER

# Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Verify installation
docker --version
```

**Important**: After adding your user to the docker group, you must either:
- Log out and log back in, OR
- Run `newgrp docker` in your terminal

This is required for the group membership to take effect.

### 3. Docker iptables Configuration (Jetson-Specific)

The Jetson kernel doesn't include the `iptable_raw` module, which Docker tries to use by default. This causes errors when running containers. Configure Docker to work around this:

```bash
# Create Docker daemon configuration directory if it doesn't exist
sudo mkdir -p /etc/docker

# Create daemon.json with iptables workaround
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "iptables": false,
  "ip-forward": true
}
EOF

# Restart Docker daemon
sudo systemctl restart docker
```

**Note**: Disabling iptables means Docker won't manage network isolation between containers. This is acceptable for development/testing. For production, consider using host networking mode (`--network host`) if needed.

### 4. NVIDIA Container Toolkit Installation

Required for GPU access in Docker containers.

```bash
# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y curl

# Configure repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install nvidia-container-toolkit packages
sudo apt-get update
sudo apt-get install -y \
    nvidia-container-toolkit \
    nvidia-container-toolkit-base \
    libnvidia-container-tools \
    libnvidia-container1

# Configure Docker to use nvidia runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker daemon
sudo systemctl restart docker

# Verify configuration
docker info | grep -i runtime
```

**Note**: The NVIDIA runtime should appear as `nvidia` in the Docker info output. If it doesn't, verify the configuration was applied correctly.

### 5. Power and performance mode (optional)

Many guides recommend setting the Jetson to maximum performance before running cameras or inference, to avoid throttling and USB/camera detection issues:

```bash
# Set power mode to max performance (mode 0; mode IDs can vary by board)
sudo nvpmodel -m 0

# Lock CPU/GPU to maximum frequencies (disables dynamic scaling)
sudo jetson_clocks
```

- **nvpmodel -m 0**: Uses the highest power profile (check available modes with `nvpmodel -q`). **Does not persist** across reboots; the board typically boots in its default (often lower) power mode.
- **jetson_clocks**: Keeps clocks at max until reboot. **Does not persist** across reboots. Omit if you prefer power saving or don’t need sustained peak performance.

Use these when you need consistent performance for the ZED camera or the locomotion pipeline; they are optional for initial testing.

**Apply on every boot (optional)** — Neither setting survives a reboot. To run both automatically at startup, create a systemd service:

```bash
sudo tee /etc/systemd/system/jetson-maxperf.service > /dev/null <<'EOF'
[Unit]
Description=Set Jetson to max performance (nvpmodel + jetson_clocks)
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/sbin/nvpmodel -m 0
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable jetson-maxperf.service
```

After the next reboot, the service will set max power mode and lock clocks. To run it once without rebooting: `sudo systemctl start jetson-maxperf.service`. To disable: `sudo systemctl disable jetson-maxperf.service`. Adjust the `-m 0` or paths if your board uses different mode IDs or install locations (`which nvpmodel jetson_clocks`).

### 6. ZED 2i Camera (Optional — Host Verification)

Optional: verify the ZED 2i on the host before using it in the container.

1. **Match L4T**: Check your L4T version (`cat /etc/nv_tegra_release`) and download the matching ZED SDK installer from [Stereolabs](https://www.stereolabs.com/developers/release/) (e.g. L4T 36.4 → `ZED_SDK_Tegra_L4T36.4_*.zstd.run`).
2. **Install**: `sudo apt-get install -y zstd` then run the `.zstd.run` installer.
3. **Verify USB**: `lsusb | grep -i zed` — device must appear on a USB 3.0 port.
4. **Run diagnostics**: `ZED_Diagnostic -c -d`

   You should see **OK** for: ZED SDK Diagnostic, Processor, Graphics Card, and CUDA Operations. Under **AI Models Diagnostic**, detection models (MULTI CLASS, HUMAN BODY, PERSON HEAD, REID) may show "not optimized" — that is normal and not used by the HAL RGB/depth pipeline. For depth, ensure **NEURAL LIGHT DEPTH**, **NEURAL DEPTH**, and **NEURAL PLUS DEPTH** show "optimized" (run the pre-optimization step below if needed).

**If the SDK can’t open the camera** (appears in `lsusb` but diagnostic fails), install udev rules:

```bash
echo ‘SUBSYSTEM==”usb”, ATTR{idVendor}==”2b03”, MODE=”0666”’ | sudo tee /etc/udev/rules.d/99-slabs.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Unplug and replug the ZED, then retry.

**Two ZED units:** the default catalog’s second RGB-D row is **`side_rgbd`** (policy side slot when `policy_scan_slot="side"`). For that row, set **`KRABBY_SIDE_ZED_USB_SERIAL`** to the **integer USB serial** of the side ZED so HAL selects the correct device; the primary `front_rgbd` ZED can use the first enumerated device when no serial is set there.

### MaixSense-A075V (optional host bring-up)

Optional RGB-D over **HTTP** (no Stereolabs SDK). Use as the primary `front_rgbd` driver or as extra `rgbd` catalog rows. Official docs: [MaixSense-A075V – Sipeed Wiki](https://wiki.sipeed.com/hardware/en/maixsense/maixsense-a075v/maixsense-a075v.html).

1. **USB / link**: Often **`0525:a4a2`** (Linux RNDIS). The module is **`192.168.233.1`** on the USB Ethernet link by default.
2. **Jetson IP**: Put **`192.168.233.2/24`** on the RNDIS interface only. Find the iface with `ip -br link` (commonly **`enx…`**, not always `usb0`). Do not attach **`192.168.233.0/24`** to another NIC.
3. **Check reachability**: `ip route get 192.168.233.1` must show the RNDIS device and `src 192.168.233.2`; then `ping -c 3 192.168.233.1` and `curl -sS -o /dev/null -w '%{http_code}\n' http://192.168.233.1/`.
4. **NetworkManager** (replace `IFACE`):

```bash
sudo nmcli connection add type ethernet ifname IFACE con-name maixsense-rndis \
  ipv4.method manual ipv4.addresses 192.168.233.2/24 \
  ipv6.method ignore connection.autoconnect yes
sudo nmcli connection up maixsense-rndis
```

5. **Web UI**: `http://192.168.233.1` (~10–15 s after power-on). Remote browser via Jetson: `ssh -N -L 8080:192.168.233.1:80 USER@JETSON` then open `http://127.0.0.1:8080`.
6. **Driver problems**: [Sipeed install / driver notes](https://wiki.sipeed.com/hardware/en/maixsense/maixsense-a075v/install_drivers.html).

**HAL wiring**

- Python extras: `pip install "krabby-hal-server-jetson[maixsense]"` (`requests`, `opencv-python-headless`).
- Catalog: set **`camera_driver="maixsense_a075v"`** on each **`rgbd`** row that uses MaixSense (primary **`front_rgbd`** and/or extra rows with **`hal_open_rgbd=True`**). Each such row must set **`maixsense_host_env`** / optional **`maixsense_port_env`** to the **names** of env vars that hold that module’s HTTP host and port—**you choose those names** in **`JETSON_SENSOR_CATALOG`** (distinct per module). Deployment passes **one `-e`** (and optional port **`-e`**) per name.
- **Policy** uses **`camera_*`** / **`scan_features`** from the primary row; optional **`side_*`** when one row has **`policy_scan_slot="side"`** and the checkpoint uses **`num_side_scan`**. **Collision / extra streams**: read **`HardwareObservations.rgbd_by_catalog_id[id].rgb`** / **`.depth`** (each row’s own resolution). Implementation: `hal/server/jetson/maixsense_a075v.py`, `maixsense_rgb_depth_camera.py`, **`JETSON_SENSOR_CATALOG`** in `sensor_backend_jetson.py`.
- **Hardware smoke test**: `scripts/run_jetson_maixsense_hal_hw_test.sh` (expects Docker image `krabby-locomotion:latest`, **`--network host`**, and **`KRABBY_MAIXSENSE_LIVE_TEST_HOST`** set to the module IP; optional **`KRABBY_MAIXSENSE_LIVE_TEST_PORT`**).

**Docker**: No ZED-style USB passthrough for HTTP; the container must reach each module’s IP (**`--network host`** on Jetson is typical). For every MaixSense row, the catalog’s **`maixsense_host_env`** / **`maixsense_port_env`** name the variables you set at **`docker run`** (**`-e NAME=value`**, one pair per module)—same pattern as multiple ZED serial envs.

**HAL front camera (camera_rgb / camera_depth)**  
The Jetson HAL fills `HardwareObservations.camera_rgb` and `camera_depth` from the **front RGB-D observation camera** defined in **`JETSON_SENSOR_CATALOG`**: the row with **`id="front_rgbd"`** and **`is_primary=True`** sets **`camera_driver`** (e.g. **`zed`** or **`maixsense_a075v`**), **resolution**, and **fps**. Optional constructor overrides: **`camera_resolution`**, **`camera_fps`**, **`camera_driver`**. **`depth_mode`** applies to **ZED** only. **Policy scan** (`scan_features`, optional **`side_*`**) comes from the configured depth streams; **every opened RGB-D row** also appears under **`rgbd_by_catalog_id`**. GStreamer IDs, ZED install, and MaixSense networking: **SENSOR_INTERFACE.md**, ZED section above, and [MaixSense-A075V (optional host bring-up)](#maixsense-a075v-optional-host-bring-up). Wire format: **HAL_GUIDE.md** and `hal/client/data_structures/hardware.py`.

## Obtaining the Docker Image

The Docker image must be available on the Jetson device. Pull it from your container registry or load it from an archive, then tag it as `krabby-locomotion:latest` (or specify the image name/tag in the `docker run` commands below).

## Running on Jetson

**Important**: Checkpoint files are not included in the Docker image. You must mount your checkpoint directory as a volume using `-v /path/to/checkpoints:/workspace/checkpoints`. All examples below assume checkpoints are mounted at `/workspace/checkpoints` inside the container.

**Optional: Enable data collection with host persistence**

Create the host folder first:

```bash
mkdir -p /path/to/krabby_bags
```

Set `--data-collector-output-dir` to enable recording. The mount target must match the same container path:

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

### Important: GPU Runtime Flag

**On Jetson, use `--runtime=nvidia` instead of `--gpus all`:**

```bash
# Correct for Jetson
docker run --rm --runtime=nvidia <image> <command>

# If you get "unknown or invalid runtime name: nvidia", reconfigure:
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Basic Usage

Run the production inference runner:

```bash
docker run --rm --runtime=nvidia \
    -v /path/to/checkpoints:/workspace/checkpoints \
    -v /dev:/dev \
    krabby-locomotion:latest \
    --checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt
```

**Note**: 
- The container entrypoint is `hal.server.jetson.main`, so you can pass arguments directly without specifying the Python module.
- Replace `/path/to/checkpoints` with the actual path to your checkpoint directory on the host.
- Ensure the checkpoint file exists at the specified path (e.g., `/path/to/checkpoints/unitree_go2_parkour_teacher.pt` must exist on the host).

### With ZED 2i Camera

The ZED SDK accesses the camera over USB (libusb). `--device /dev/video0` alone is not enough; the container requires `-v /dev:/dev` and `--privileged`:

```bash
docker run --rm --runtime=nvidia \
    -v /path/to/checkpoints:/workspace/checkpoints \
    -v /dev:/dev \
    --privileged \
    krabby-locomotion:latest \
    --checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt
```

**Note**: On first use the ZED SDK downloads and optimizes NEURAL depth models, adding several minutes to startup. Pre-optimize using the section below to avoid this.

### With MaixSense-A075V (HTTP in container)

Configure RNDIS on the **Jetson host** first ([MaixSense-A075V (optional host bring-up)](#maixsense-a075v-optional-host-bring-up)). The container must reach **every** module IP you use; **`--network host`** is typical.

**Example (`docker run`):** use **`--network host`** so the container can reach each module. Each **`-e`** name must match that row’s **`maixsense_host_env`** (and port env if **`maixsense_port_env`** is set) in **`JETSON_SENSOR_CATALOG`**. Below uses illustrative host env names for a **front** + **side** pair; rename them in the catalog and in **`docker run`** if you prefer other strings.

```bash
docker run --rm --runtime=nvidia --network host \
    -v /path/to/checkpoints:/workspace/checkpoints \
    -e KRABBY_JETSON_MAIXSENSE_FRONT_HOST=192.168.233.1 \
    -e KRABBY_JETSON_MAIXSENSE_SIDE_HOST=192.168.234.1 \
    krabby-locomotion:latest \
    --checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt
```

Add more **`-e …`** (and optional **`-e …_PORT=…`**) for every MaixSense row you open. Third-party USB–Ethernet links may use different subnets—set values accordingly.

### Pre-optimizing ZED NEURAL depth models

Optimization is GPU-specific and must run on the target Jetson. Use a persistent volume to avoid re-downloading on every run:

1. Run the image once to download and optimize the models:

   ```bash
   yes | docker run --rm --runtime=nvidia -v /dev:/dev --privileged \
       -v ~/zed-resources:/usr/local/zed/resources \
       -i krabby-locomotion:latest \
       bash -c "ZED_Diagnostic -nrlo && ZED_Diagnostic -nrlo_plus"
   ```

   This takes several minutes. The models are saved to `~/zed-resources` on the host.

2. Mount the same directory on subsequent runs:

   ```bash
   docker run --rm --runtime=nvidia \
       -v /path/to/checkpoints:/workspace/checkpoints \
       -v /dev:/dev --privileged \
       -v ~/zed-resources:/usr/local/zed/resources \
       krabby-locomotion:latest \
       --checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt
   ```

### In-Process Mode (Default - Recommended for Production)

By default, the inference runner uses in-process communication (`inproc://`) which is more efficient for single-container deployment:

```bash
docker run --rm --runtime=nvidia \
    -v /path/to/checkpoints:/workspace/checkpoints \
    -v /dev:/dev \
    krabby-locomotion:latest \
    --checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt
```

**Note**: In-process mode (`inproc://`) provides zero-copy communication and lowest latency. This is the recommended mode for production deployment.

### Network Mode (Cross-Container Communication)

For running HAL server and client in separate containers:

**HAL Server Container:**
```bash
docker run --rm --runtime=nvidia \
    -v /path/to/checkpoints:/workspace/checkpoints \
    -v /dev:/dev \
    -p 6001:6001 -p 6002:6002 \
    --name hal-server \
    krabby-locomotion:latest \
    --checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt \
    --observation_bind tcp://*:6001 \
    --command_bind tcp://*:6002
```

**HAL Client Container (on same or different machine):**
```bash
docker run --rm --runtime=nvidia \
    -v /path/to/checkpoints:/workspace/checkpoints \
    --name hal-client \
    krabby-locomotion:latest \
    --checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt \
    --observation_endpoint tcp://hal-server:6001 \
    --command_endpoint tcp://hal-server:6002
```

**Note**: Network mode is useful for debugging and multi-container setups, but has higher latency than in-process mode. Both containers require checkpoint volume mounts.

## Environment Variables

- `CUDA_VISIBLE_DEVICES`: Control which GPU to use (default: all)
- `HAL_BASE_PORT`: Base port for HAL endpoints (default: 6000)
- `LOG_LEVEL`: Logging level (default: INFO)

## Troubleshooting

### Permission Denied Errors

If you see "permission denied while trying to connect to the docker API":
1. Verify you're in the docker group: `groups`
2. If not, run `newgrp docker` or log out/in
3. Verify docker socket permissions: `ls -l /var/run/docker.sock` (should show `docker` group)

### NVIDIA Runtime Not Found

If you get "unknown or invalid runtime name: nvidia":
1. Verify nvidia-container-toolkit is installed: `which nvidia-ctk`
2. Reconfigure: `sudo nvidia-ctk runtime configure --runtime=docker`
3. Restart Docker: `sudo systemctl restart docker`
4. Verify: `docker info | grep -i runtime`

### Docker iptables Errors

If containers fail with iptables errors:
1. Check `/etc/docker/daemon.json` contains `"iptables": false`
2. Restart Docker: `sudo systemctl restart docker`
3. Note: This disables Docker's network isolation (acceptable for development)

### CUDA Out of Memory

If you encounter CUDA OOM errors:
- Reduce batch size if applicable
- Use TensorRT optimization (see `export_to_tensorrt.py`)
- Ensure no other processes are using GPU memory
- Check GPU memory usage: `nvidia-smi`

### ZED Camera Not Detected

- Verify camera is connected: `lsusb | grep ZED`
- The ZED SDK uses USB (not only V4L2). Ensure the container is run with `-v /dev:/dev` and `--privileged` so it can access `/dev/bus/usb`. `--device /dev/video0` alone is often insufficient.

### High Latency

- Verify GPU is being used: Check logs for "CUDA available" and device name
- Consider TensorRT export for optimization
- Check system load and thermal throttling: `tegrastats`
- Ensure using `--runtime=nvidia` for GPU access

## Performance Tuning

### TensorRT Optimization

For faster inference, export the model to TensorRT format. This requires first exporting to ONNX, then converting to TensorRT:

1. **Export to ONNX**: Use the training scripts to export the model to ONNX format (see `parkour/scripts/rsl_rl/play.py` for ONNX export functionality).

2. **Convert ONNX to TensorRT**: Use TensorRT's `trtexec` tool or Python API to convert the ONNX model to TensorRT engine format.

**Note**: TensorRT optimization can significantly reduce inference latency on Jetson devices. The exact conversion process depends on your model architecture and TensorRT version. Refer to NVIDIA's TensorRT documentation for detailed conversion steps.

### Control Loop Rate

The HAL runtime currently uses a fixed in-code control loop rate of 100 Hz.

## Production Deployment Notes

### System Configuration

- **JetPack Version**: 6.1/6.2 (L4T 36.4) or later recommended
- **Docker**: Must be configured with iptables workaround for Jetson kernel
- **NVIDIA Runtime**: Must be configured for GPU access (`--runtime=nvidia`)
- **User Permissions**: User must be in docker group to run containers without sudo

### Performance Considerations

- **GPU Usage**: Always use `--runtime=nvidia` for GPU-accelerated inference
- **Communication Mode**: Use in-process mode (`inproc://`) for production (lowest latency)
- **Camera**: ZED camera initialization is required for production (not optional)
- **TensorRT**: Consider TensorRT export for optimized inference performance

### Important Notes

- **Network Isolation**: With `iptables: false`, Docker won't manage network isolation between containers. Use host networking mode if needed.
- **Camera**: ZED camera must be initialized for production deployment to provide valid depth/scan features for inference.

## Isaac Sim synthetic front camera

When running the HAL server in Isaac Sim, the **synthetic front camera** is used: the first camera in `scene.sensors` that provides depth/RGB is used to fill `camera_rgb` and `camera_depth` (same as `rgb_camera_1` and `depth_map`). Add a camera in the scene that approximates the real ZED (position and FOV) and attach it to the robot or a fixed frame. Resolution is 480×640 in the current implementation. Validate by connecting a HAL client to the Isaac HAL observation endpoint and verifying `camera_rgb` and `camera_depth` in the observations.

## Additional Resources

- Docker dependencies: See `docs/DOCKER_DEPENDENCIES.md`
- HAL architecture and observation format: See `docs/HAL_GUIDE.md`

