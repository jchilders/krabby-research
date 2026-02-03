#!/usr/bin/env bash
# Run the gamepad-only HAL server (server only) inside the locomotion container.
# Requires: krabby-locomotion image built (make build-locomotion-image on Orin).
# Usage: ./controller/scripts/jetson/helper/run_gamepad_hal_server_only_in_container.sh [--mcu-port PORT] [--mcu-baud BAUD]
# Then run the client on the host: python controller/scripts/jetson/run_gamepad_to_krabby_client.py

set -e
IMAGE="${KRABBY_LOCOMOTION_IMAGE:-krabby-locomotion:latest}"
docker run --rm --privileged --runtime=nvidia \
  -v /dev:/dev \
  -p 6001:6001 -p 6002:6002 \
  --entrypoint python3 \
  "$IMAGE" \
  -m controller.scripts.jetson.main_gamepad_only \
  --observation_bind tcp://*:6001 --command_bind tcp://*:6002 \
  "$@"
