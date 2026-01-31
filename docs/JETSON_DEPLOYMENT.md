# Jetson Deployment Guide

This guide explains how to build and deploy the parkour policy runtime on Jetson Orin hardware.

## System Requirements

- **Device**: NVIDIA Jetson Orin
- **OS**: Ubuntu 22.04.5 LTS (Jammy Jellyfish)
- **Kernel**: 5.15.148-tegra (NVIDIA Tegra)
- **Architecture**: aarch64 (ARM64)
- **JetPack**: 6.1/6.2 (L4T 36.4) or later
- **ZED Camera**: Optional, for depth sensing (ZED SDK 5.1.1 for L4T 36.4)
- **Model checkpoint file**: `.pt` format (e.g., `unitree_go2_parkour_teacher.pt`)

## Prerequisites: System Setup

Before building and deploying, ensure the Jetson system is properly configured.

### 1. Docker Installation

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

### 2. Docker iptables Configuration (Jetson-Specific)

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

### 3. NVIDIA Container Toolkit Installation

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
    --action_dim 12 \
    --obs_dim 753 \
    --inference_device cuda \
    --control_rate 100.0
```

**Note**: 
- The container entrypoint is `hal.server.jetson.main`, so you can pass arguments directly without specifying the Python module.
- Replace `/path/to/checkpoints` with the actual path to your checkpoint directory on the host.
- Ensure the checkpoint file exists at the specified path (e.g., `/path/to/checkpoints/unitree_go2_parkour_teacher.pt` must exist on the host).

### With ZED Camera

If using ZED Mini camera, mount the camera device:

```bash
docker run --rm --runtime=nvidia \
    -v /path/to/checkpoints:/workspace/checkpoints \
    -v /dev:/dev \
    --device /dev/video0 \
    krabby-locomotion:latest \
    --checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt \
    --action_dim 12 \
    --obs_dim 753 \
    --inference_device cuda \
    --control_rate 100.0
```

**Note**: The ZED SDK is installed automatically in the Docker image during build. The camera will be detected automatically if connected.

### In-Process Mode (Default - Recommended for Production)

By default, the inference runner uses in-process communication (`inproc://`) which is more efficient for single-container deployment:

```bash
docker run --rm --runtime=nvidia \
    -v /path/to/checkpoints:/workspace/checkpoints \
    -v /dev:/dev \
    krabby-locomotion:latest \
    --checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt \
    --action_dim 12 \
    --obs_dim 753 \
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
    --action_dim 12 \
    --obs_dim 753 \
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
    --action_dim 12 \
    --obs_dim 753 \
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
- Check device permissions: `ls -l /dev/video*`
- Try running with `--device /dev/video0` explicitly

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

