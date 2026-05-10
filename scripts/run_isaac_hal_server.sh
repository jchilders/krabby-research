#!/bin/bash
# Isaac Sim HAL + joystick demo in Docker. Build: make build-isaacsim-image
#
#   ./scripts/run_isaac_hal_server.sh              # Go2 play + krabby-uno-sim on :5555 / :5556
#   ./scripts/run_isaac_hal_server.sh --hexapod  # Hex USD from repo assets/
#
# Optional flags (--teleop, --hexapod) may appear anywhere among script arguments.
# Everything else is passed through to the container entrypoint.
# Teleop signaling is configured inside the image (teleop/edge/robot_settings.py).

set -e

IMAGE="krabby-isaacsim:latest"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Demo pins (edit here if the demo changes)
GO2_TASK="Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0"
HEX_USD="/workspace/assets/crab_hex_ref.usd"

want_teleop=0
want_hexapod=0
passthrough=()
for arg in "$@"; do
  if [ "$arg" = "--teleop" ]; then
    want_teleop=1
  elif [ "$arg" = "--hexapod" ]; then
    want_hexapod=1
  else
    passthrough+=("$arg")
  fi
done

py=(--joystick)

if [ "$want_teleop" -eq 1 ]; then
  py+=(--teleop)
fi

if [ "$want_hexapod" -eq 1 ]; then
  py+=(--usd "$HEX_USD")
else
  py+=(--task "$GO2_TASK")
  if [ "$want_teleop" -eq 1 ]; then
    # Teleop attaches sim cameras and requires --robot (not auto); Go2 play uses go2.
    py+=(--robot go2)
  fi
fi

py+=("${passthrough[@]}")

xhost +local:docker 2>/dev/null || true

if [ "$want_hexapod" -eq 1 ]; then
  exec docker run --rm --gpus all \
    -p 5555:5555 -p 5556:5556 \
    -e "DISPLAY=${DISPLAY}" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "${REPO_ROOT}/assets:/workspace/assets" \
    "$IMAGE" "${py[@]}"
else
  exec docker run --rm --gpus all \
    -p 5555:5555 -p 5556:5556 \
    -e "DISPLAY=${DISPLAY}" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    "$IMAGE" "${py[@]}"
fi
