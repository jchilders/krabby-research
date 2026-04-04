#!/bin/bash
# Run the Isaac Sim HAL server in joystick mode inside the Docker image.
# Build the image first: make build-isaacsim-image (from krabby-research).
# Extra arguments are passed to the container (e.g. --video, --num_envs 8).
#
# Usage:
#   ./scripts/run_isaac_hal_server.sh              # Go2 parkour task
#   ./scripts/run_isaac_hal_server.sh --hexapod   # Crab hex from assets/crab_hex_ref.usd

set -e

IMAGE="${KRABBY_ISAACSIM_IMAGE:-krabby-isaacsim:latest}"
xhost +local:docker 2>/dev/null || true

if [ "${1:-}" = "--hexapod" ]; then
  shift
  # Mount assets so the container can load crab_hex_ref.usd; use --usd (sets task to Isaac-CrabHex-Joystick-v0, robot hex).
  REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  exec docker run --rm --gpus all -p 5555:5555 -p 5556:5556 \
    -e DISPLAY="${DISPLAY}" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "${REPO_ROOT}/assets:/workspace/assets" \
    "$IMAGE" \
    --joystick --usd /workspace/assets/crab_hex_ref.usd \
    "$@"
else
  exec docker run --rm --gpus all -p 5555:5555 -p 5556:5556 \
    -e DISPLAY="${DISPLAY}" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    "$IMAGE" \
    --joystick --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 \
    "$@"
fi
