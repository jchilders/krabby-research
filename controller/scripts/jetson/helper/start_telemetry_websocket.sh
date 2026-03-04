#!/usr/bin/env bash
# Run Jetson HAL websocket telemetry in telemetry-only mode (no inference loop).
#
# Defaults:
# - No NVIDIA runtime flag (works when Docker runtime "nvidia" is not configured)
# - Host network + /dev passthrough for MCU serial access
#
# Environment overrides:
#   KRABBY_LOCOMOTION_IMAGE     (default: krabby-locomotion:latest)
#   KRABBY_TELEMETRY_TOKEN      (required)
#   KRABBY_MCU_PORT             (default: /dev/ttyACM0)
#   KRABBY_TELEMETRY_WS_HOST    (default: 0.0.0.0)
#   KRABBY_TELEMETRY_WS_PORT    (default: 8787)
#   KRABBY_TELEMETRY_WS_PATH    (default: /krabby/telemetry)
#   KRABBY_TELEMETRY_WS_HZ      (default: 10)
#   KRABBY_TELEMETRY_FAKE_DATA  (default: 0; set to 1 to stream synthetic telemetry)
#   KRABBY_USE_NVIDIA_RUNTIME   (default: 0; set to 1 to add --runtime=nvidia)
#   KRABBY_DOCKER_NETWORK_MODE  (default: auto; host on Linux, bridge elsewhere)
#
# Script flags:
#   --dry-run                   Print the docker command and exit
#   -h, --help                  Show help
#   --                          Pass remaining args to the HAL entrypoint

set -euo pipefail

DRY_RUN=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage: ./controller/scripts/jetson/helper/start_telemetry_websocket.sh [--dry-run] [-- <extra hal args>]

Starts Jetson HAL websocket telemetry in telemetry-only mode.
Set KRABBY_TELEMETRY_TOKEN before running.
Set KRABBY_TELEMETRY_FAKE_DATA=1 to stream synthetic telemetry.
EOF
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        EXTRA_ARGS+=("$1")
        shift
      done
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

IMAGE="${KRABBY_LOCOMOTION_IMAGE:-krabby-locomotion:latest}"
TOKEN="${KRABBY_TELEMETRY_TOKEN:-}"
MCU_PORT="${KRABBY_MCU_PORT:-/dev/ttyACM0}"
WS_HOST="${KRABBY_TELEMETRY_WS_HOST:-0.0.0.0}"
WS_PORT="${KRABBY_TELEMETRY_WS_PORT:-8787}"
WS_PATH="${KRABBY_TELEMETRY_WS_PATH:-/krabby/telemetry}"
WS_HZ="${KRABBY_TELEMETRY_WS_HZ:-10}"
FAKE_DATA="${KRABBY_TELEMETRY_FAKE_DATA:-0}"
USE_NVIDIA_RUNTIME="${KRABBY_USE_NVIDIA_RUNTIME:-0}"
DOCKER_NETWORK_MODE="${KRABBY_DOCKER_NETWORK_MODE:-auto}"

if [[ -z "$TOKEN" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    TOKEN="<set-KRABBY_TELEMETRY_TOKEN>"
    echo "Warning: KRABBY_TELEMETRY_TOKEN is not set; using placeholder token in dry run." >&2
  else
    echo "Error: KRABBY_TELEMETRY_TOKEN is required." >&2
    echo "Example: export KRABBY_TELEMETRY_TOKEN=dev-token" >&2
    exit 1
  fi
fi

if [[ "$FAKE_DATA" != "1" && ! -e "$MCU_PORT" ]]; then
  echo "Warning: MCU device not found at $MCU_PORT. Telemetry stream will start in disconnected mode." >&2
  echo "Set KRABBY_MCU_PORT to the correct serial device if needed." >&2
fi

echo "Starting telemetry websocket from image: $IMAGE"
echo "Endpoint: ws://$WS_HOST:$WS_PORT$WS_PATH"
if [[ "$FAKE_DATA" == "1" ]]; then
  echo "Mode: fake telemetry enabled"
fi

run_cmd=(
  docker run --rm --privileged
  -v /dev:/dev
  -e "KRABBY_MCU_PORT=$MCU_PORT"
)

if [[ "$DOCKER_NETWORK_MODE" == "auto" ]]; then
  if [[ "$(uname -s)" == "Linux" ]]; then
    DOCKER_NETWORK_MODE="host"
  else
    DOCKER_NETWORK_MODE="bridge"
  fi
fi

if [[ "$DOCKER_NETWORK_MODE" == "host" ]]; then
  echo "Docker network mode: host"
  run_cmd+=(--network host)
else
  echo "Docker network mode: bridge (publishing $WS_PORT:$WS_PORT)"
  run_cmd+=(-p "$WS_PORT:$WS_PORT")
fi
if [[ "$USE_NVIDIA_RUNTIME" == "1" ]]; then
  run_cmd+=(--runtime=nvidia)
fi

run_cmd+=(
  "$IMAGE"
  --telemetry_only
  --telemetry_ws_enabled
  --telemetry_ws_host "$WS_HOST"
  --telemetry_ws_port "$WS_PORT"
  --telemetry_ws_path "$WS_PATH"
  --telemetry_ws_publish_hz "$WS_HZ"
  --telemetry_ws_token "$TOKEN"
)
if [[ "$FAKE_DATA" == "1" ]]; then
  run_cmd+=(--telemetry_ws_fake_data)
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  run_cmd+=("${EXTRA_ARGS[@]}")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  printf "Dry run command:\n"
  printf "  "
  printf "%q " "${run_cmd[@]}"
  printf "\n"
  exit 0
fi

exec "${run_cmd[@]}"
