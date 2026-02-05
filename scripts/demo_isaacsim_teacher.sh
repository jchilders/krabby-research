#!/bin/bash
# Script to run Isaac Sim Docker image with teacher checkpoint and task
# Based on parkour/README.md and images/isaacsim/README.md

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get project root (parent directory of scripts folder)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Verify we're in the right place (check for Makefile)
if [ ! -f "$PROJECT_ROOT/Makefile" ]; then
    echo "Error: Could not find Makefile at $PROJECT_ROOT/Makefile"
    echo "Please ensure this script is in the scripts/ folder of the krabby-research repository."
    exit 1
fi

# Change to project root for make command
cd "$PROJECT_ROOT"

# Configure X11 access for display forwarding
xhost +local:docker 2>/dev/null

# Build the Isaac Sim Docker image
make build-isaacsim-image

# Run Isaac Sim container with teacher policy
docker run --rm --gpus all \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    krabby-isaacsim:latest \
    --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 \
    --checkpoint /workspace/test_assets/checkpoints/unitree_go2_parkour_teacher.pt \
    "$@"

