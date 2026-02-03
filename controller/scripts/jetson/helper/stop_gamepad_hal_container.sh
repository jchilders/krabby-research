#!/usr/bin/env bash
# Stop the hal-gamepad container (server + client-in-container flow).
# Run this on the host after pressing Ctrl+C in the client terminal.
# Usage: ./controller/scripts/jetson/helper/stop_gamepad_hal_container.sh

set -e
docker stop hal-gamepad
