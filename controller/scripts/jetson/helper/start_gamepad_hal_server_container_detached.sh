#!/usr/bin/env bash
# Start the gamepad-only HAL server container in detached mode so you can run the client inside it.
# Then run ./controller/scripts/jetson/helper/run_gamepad_client_in_container.sh in another terminal.
# Stop with ./controller/scripts/jetson/helper/stop_gamepad_hal_container.sh
# Usage: ./controller/scripts/jetson/helper/start_gamepad_hal_server_container_detached.sh [--mcu-port PORT] [--mcu-baud BAUD]

set -e
IMAGE="${KRABBY_LOCOMOTION_IMAGE:-krabby-locomotion:latest}"
docker run -d --name hal-gamepad --rm --privileged --runtime=nvidia \
  -v /dev:/dev \
  -p 6001:6001 -p 6002:6002 \
  --entrypoint python3 \
  "$IMAGE" \
  -m controller.scripts.jetson.main_gamepad_only \
  --observation_bind tcp://*:6001 --command_bind tcp://*:6002 \
  "$@"
