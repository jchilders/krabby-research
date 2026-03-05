#!/bin/bash
# Run the Isaac Sim HAL server in joystick mode inside the Docker image.
# Build the image first: make build-isaacsim-image (from krabby-research).
# Extra arguments are passed to the container (e.g. --video, --num_envs 8).

set -e

IMAGE="${KRABBY_ISAACSIM_IMAGE:-krabby-isaacsim:latest}"
xhost +local:docker 2>/dev/null || true

exec docker run --rm --gpus all -p 5555:5555 -p 5556:5556 \
  -e DISPLAY="${DISPLAY}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  "$IMAGE" \
  --joystick --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 \
  "$@"
