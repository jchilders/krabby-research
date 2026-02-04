#!/usr/bin/env bash
# Stop the hal-gamepad container (stops both HAL server and client when using
# start_gamepad_hal_and_client_detached.sh, or the server when using the two-terminal flow).
# Usage: ./controller/scripts/jetson/helper/stop_gamepad_hal_container.sh

set -e
docker stop hal-gamepad
