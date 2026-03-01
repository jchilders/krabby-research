# Jetson Deployment Guide

This guide explains how to build and deploy the parkour policy runtime on Jetson Orin hardware.

## System Requirements

- **Device**: NVIDIA Jetson Orin
- **OS**: Ubuntu 22.04.5 LTS (Jammy Jellyfish)
- **Kernel**: 5.15.148-tegra (NVIDIA Tegra)
- **Architecture**: aarch64 (ARM64)
- **JetPack**: 6.1/6.2 (L4T 36.4) or later
- **ZED 2i Camera**: Optional, for depth sensing (ZED SDK 5.1.1+ for L4T 36.4)
- **Model checkpoint file**: `.pt` format (e.g., `unitree_go2_parkour_teacher.pt`)

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

Optional: verify the ZED 2i on the host before using it in the container. The Docker image includes the ZED SDK; production uses the camera from inside the container.

- **L4T match**: Use a ZED SDK build that matches L4T (check with `cat /etc/nv_tegra_release`). Get the installer from [Stereolabs ZED SDK](https://www.stereolabs.com/developers/release/) for your L4T (e.g. L4T 36.4 → `ZED_SDK_Tegra_L4T36.4_*.zstd.run`).
- **Install**: `sudo apt-get install -y zstd` then run the downloaded `.zstd.run` installer; accept the license.
- **Verify USB**: `lsusb | grep -i zed` — device must appear. Use a USB 3.0 port; if missing, try `sudo nvpmodel -m 0` and `sudo jetson_clocks` (see [§5](#5-power-and-performance-mode-optional)).

#### 6.2 Run ZED Explorer for verification

ZED Explorer is a GUI application. Use one of these options:

**Option A — X11 forwarding over SSH (no physical display)**

From your local machine, connect with X11 forwarding:

```bash
ssh -X jetson
# Or: ssh -X <username>@<hostname-or-ip>
```

On the Jetson, run:

```bash
ZED_Explorer
```

Ensure an X server is running on your local machine (e.g. XQuartz on macOS, VcXsrv or Windows X Server on Windows, or a native X/Wayland session on Linux). If you get “cannot open display”, X11 forwarding is not in use or your local DISPLAY is not set.

**Option B — Direct display on the Jetson**

Connect a monitor, keyboard, and mouse to the Jetson. Log in locally (or over SSH and set `DISPLAY=:0` if a user is logged in on the console). Then run:

```bash
ZED_Explorer
```

#### 6.3 ZED Diagnostic (CLI troubleshooting)

**ZED_Diagnostic** runs hardware and system checks without a GUI. Useful when you don’t have a display or want quick feedback in the terminal.

```bash
# Command-line mode, auto-run diagnostics (camera, GPU, SDK, CUDA)
ZED_Diagnostic -c -d
```

Other useful flags (run `ZED_Diagnostic -h` for full list):

- `-c` — Command-line mode (no GUI)
- `-d` — Auto-start diagnostics
- `-nrlo` — Download and optimize NEURAL depth model
- `-nrlo_plus` — NEURAL PLUS depth model
- `-aio` — Download and optimize all AI models

If the camera is detected, the diagnostic will report device info and run tests. If it fails to open the camera, the output helps narrow down USB vs SDK vs driver issues.

**Troubleshooting**

- **Camera not in lsusb**: Try another USB 3.0 port; if the board is in low power mode, run `sudo nvpmodel -m 0` and `sudo jetson_clocks` (see [§5](#5-power-and-performance-mode-optional)).
- **ZED appears in lsusb but ZED Explorer stays on “waiting for camera”**: The kernel sees the USB device, but the SDK cannot open it—usually due to **udev permissions**. You can run `sudo ZED_Explorer` as a quick test (root can access the device); for a proper fix, install udev rules so the ZED (vendor ID `2b03`) is accessible without root:
  1. **Option A — From the ZED SDK installer** (if you have the `.run` file): extract and install the rules:  
     `bash ./ZED_SDK_*.run --tar -x 99-slabs.rules`  
     (The file is extracted into the current directory; ignore tar “Not found” messages if `99-slabs.rules` appears.) Then:  
     `sudo mv 99-slabs.rules /etc/udev/rules.d/`  
     `sudo udevadm control --reload-rules && sudo udevadm trigger`
  2. **Option B — Manual udev rule**: create `/etc/udev/rules.d/99-slabs.rules` with:
     ```
     # Stereolabs ZED (2b03:f880 ZED 2i, 2b03:f881 HID interface)
     SUBSYSTEM=="usb", ATTR{idVendor}=="2b03", MODE="0666"
     ```
     Then run:  
     `sudo udevadm control --reload-rules && sudo udevadm trigger`  
  Unplug and replug the ZED, or reboot (`sudo reboot`), then try ZED Explorer again.
- **Still “waiting for camera” after udev and reboot:**  
  1. **Confirm it’s permissions**: Run `sudo ZED_Explorer`. If it works with sudo, the SDK still can’t open the device as your user.  
  2. **Override with a simple rule**: The installer’s `99-slabs.rules` may use a group your user isn’t in. Replace or add a rule that opens the device to everyone:  
     `echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="2b03", MODE="0666"' | sudo tee /etc/udev/rules.d/99-slabs.rules`  
     Then `sudo udevadm control --reload-rules && sudo udevadm trigger`, unplug and replug the ZED, and try again without sudo.  
  3. **Check device permissions**: After replug, run `ls -l /dev/bus/usb/001/` and find the two ZED devices (e.g. `008` and `009`). They should show `crw-rw-rw-` (0666) or your user in the group. If they’re `root:root` and `crw-r--r--`, the rule didn’t apply.  
  4. **Check kernel/driver messages**: Run `dmesg -w`, then unplug and replug the ZED. Look for `uvcvideo` or `usb` errors or “Permission denied”.  
  5. **SDK vs L4T**: Ensure your ZED SDK build matches L4T (e.g. 5.2 for L4T 36.4). Mismatches can cause the SDK to fail to open the camera even when USB is visible. See [Stereolabs ZED SDK releases](https://www.stereolabs.com/developers/release/) and the [Stereolabs forum](https://community.stereolabs.com/) for your board and JetPack version.
- **ZED_Diagnostic segfaults (Segmentation fault / core dumped) during or after the USB camera step:** The diagnostic can crash on Jetson when probing the USB camera (known with some SDK/JetPack combinations). Try: (1) **Connect the ZED directly** to a Jetson USB 3.0 port (no hub). (2) **Skip the diagnostic** and test with `ZED_Explorer` or your application—the camera may work even if the diagnostic crashes. (3) If you need a working diagnostic, try another ZED SDK version (e.g. 5.1.x) matching your L4T. (4) Report to [Stereolabs support](https://community.stereolabs.com/) with your JetPack/L4T version, ZED SDK version, and the stack trace.
- **ZED_Diagnostic reports “USB Camera Diagnostic: Failed – No Camera detected” but lsusb shows the ZED (and Argus errors appear):** The SDK may be failing to open the UVC device or may be affected by Jetson’s camera stack. Try: (1) **Connect the ZED directly** to a Jetson USB 3.0 port (no hub)—some hubs cause enumeration or descriptor issues. (2) **Check the video device**: run `ls -l /dev/video*` and `v4l2-ctl --list-devices`; the ZED 2i (2b03:f880) should appear as a UVC device (e.g. `/dev/video0`). If there is no video node or permissions are wrong, the SDK cannot open it. (3) **Restart the Argus daemon**: `sudo systemctl restart nvargus-daemon` (sometimes needed after hotplug). (4) If it still fails, the Argus “EndOfFile” messages may be a known Jetson/USB interaction—report to [Stereolabs support](https://community.stereolabs.com/) or [support@stereolabs.com](mailto:support@stereolabs.com) with your JetPack/L4T and ZED SDK versions.
- **Permission denied on device**: Ensure your user is in the appropriate group (e.g. `plugdev`) or run with `sudo` only for testing. If the problem persists, install the udev rules above.
- **ZED_Explorer not found**: Ensure the ZED SDK installer completed and that your shell’s `PATH` includes the ZED install directory (the installer usually adds it; log out and back in if needed).

## Obtaining the Docker Image

The Docker image must be available on the Jetson device. Pull it from your container registry or load it from an archive, then tag it as `krabby-locomotion:latest` (or specify the image name/tag in the `docker run` commands below).

## Running on Jetson

**Important**: Checkpoint files are not included in the Docker image. You must mount your checkpoint directory as a volume using `-v /path/to/checkpoints:/workspace/checkpoints`. All examples below assume checkpoints are mounted at `/workspace/checkpoints` inside the container.

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
    --checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt \
    --inference_device cuda \
    --control_rate 100.0
```

**Note**: 
- The container entrypoint is `hal.server.jetson.main`, so you can pass arguments directly without specifying the Python module.
- Replace `/path/to/checkpoints` with the actual path to your checkpoint directory on the host.
- Ensure the checkpoint file exists at the specified path (e.g., `/path/to/checkpoints/unitree_go2_parkour_teacher.pt` must exist on the host).

### With ZED 2i Camera

The ZED SDK talks to the camera over **USB** (libusb), not only V4L2. The container needs access to the USB device (e.g. under `/dev/bus/usb`). Stereolabs recommend `--privileged` and `-v /dev:/dev` for ZED in Docker; `--device /dev/video0` alone is usually **not** enough.

If using the ZED 2i camera, run with full device access:

```bash
docker run --rm --runtime=nvidia \
    -v /path/to/checkpoints:/workspace/checkpoints \
    -v /dev:/dev \
    --privileged \
    krabby-locomotion:latest \
    --checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt \
    --inference_device cuda \
    --control_rate 100.0
```

**Note**: The ZED SDK is installed automatically in the Docker image during build using **silent mode** (license is accepted non-interactively). The camera will be detected automatically if connected. `-v /dev:/dev` exposes all host devices (including `/dev/bus/usb`); `--privileged` allows the container to open them.

**NEURAL depth models**: The ZED SDK downloads and optimizes NEURAL/NEURAL PLUS depth models on first use, which can add several minutes to startup and will prompt *"Do you want to download and optimize the NEURAL Depth models now? [Y/n]"* if run interactively. To avoid that delay and prompt on every deployment, see [Pre-optimizing ZED NEURAL depth models](#pre-optimizing-zed-neural-depth-models) below.

### Pre-optimizing ZED NEURAL depth models

Optimization is **GPU-specific** and must run on the target Jetson. To bake optimized models into your workflow so they are reused and you don’t wait (or get prompted) at first runtime:

**Option A — One-time setup with a persistent volume (recommended)**

1. Create a host directory (or Docker named volume) to hold ZED resources, e.g. `~/zed-resources`.
2. Run the image once with GPU and device access, mounting that directory over the ZED resources path, and run the diagnostic to download and optimize NEURAL (and optionally NEURAL PLUS) depth models:

   ```bash
   docker run --rm --runtime=nvidia -v /dev:/dev --privileged \
       -v ~/zed-resources:/usr/local/zed/resources \
       krabby-locomotion:latest \
       bash -c "ZED_Diagnostic -nrlo && ZED_Diagnostic -nrlo_plus"
   ```

   If the tool prompts *"Do you want to download and optimize... [Y/n]"*, answer `Y`. To run fully non-interactively (e.g. from a script), pipe `yes` from the host: `yes | docker run --rm --runtime=nvidia -v /dev:/dev --privileged -v ~/zed-resources:/usr/local/zed/resources -i krabby-locomotion:latest bash -c "ZED_Diagnostic -nrlo && ZED_Diagnostic -nrlo_plus"`. Optimization can take several minutes.
3. When running your app, mount the same directory so the SDK uses the pre-optimized models:

   ```bash
   docker run --rm --runtime=nvidia \
       -v /path/to/checkpoints:/workspace/checkpoints \
       -v /dev:/dev --privileged \
       -v ~/zed-resources:/usr/local/zed/resources \
       krabby-locomotion:latest \
       --checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt \
       --inference_device cuda --control_rate 100.0
   ```

**Option B — Build image on Jetson and optimize in a post-build step**

If you build the image on the Jetson (not cross-build from x86), you can run a one-off container that performs the optimization and then commit the updated `/usr/local/zed` (or just the resources) into a new image layer. Because Docker build does not expose the GPU to `RUN` by default, the optimization cannot be done in a normal `RUN` in the Dockerfile; it must be done in a container that has `--runtime=nvidia`, then commit. Alternatively, use a [BuildKit GPU mount](https://github.com/NVIDIA/nvidia-container-toolkit/blob/main/docs/buildkit.md) if your setup supports it.

### In-Process Mode (Default - Recommended for Production)

By default, the inference runner uses in-process communication (`inproc://`) which is more efficient for single-container deployment:

```bash
docker run --rm --runtime=nvidia \
    -v /path/to/checkpoints:/workspace/checkpoints \
    -v /dev:/dev \
    krabby-locomotion:latest \
    --checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt \
    --inference_device cuda \
    --control_rate 100.0
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
    --inference_device cuda \
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
    --command_endpoint tcp://hal-server:6002 \
    --inference_device cuda
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

### Control Rate Adjustment

Adjust control rate based on actual latency:

```bash
# If latency is consistently < 10ms, can run at 100 Hz
--control_rate 100.0

# If latency is 10-15ms, reduce to 80 Hz
--control_rate 80.0
```

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

## Additional Resources

- Docker dependencies: See `docs/DOCKER_DEPENDENCIES.md`

