#!/usr/bin/env bash
# Run the gamepad client inside the hal-gamepad container. Start the server first with start_gamepad_hal_server_container_detached.sh.
# Usage: ./controller/scripts/jetson/helper/run_gamepad_client_in_container.sh [--device-id N] [--rate Hz]

set -e
if ! docker ps -q -f name=^hal-gamepad$ | grep -q .; then
  echo "Error: container 'hal-gamepad' is not running. Start it with ./controller/scripts/jetson/helper/start_gamepad_hal_server_container_detached.sh" >&2
  exit 1
fi
docker exec -it hal-gamepad python3 -m controller.scripts.jetson.run_gamepad_to_krabby_client \
  --observation_endpoint tcp://localhost:6001 --command_endpoint tcp://localhost:6002 \
  "$@"
