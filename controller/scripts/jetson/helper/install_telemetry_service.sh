#!/usr/bin/env bash
# Install/start a systemd service that runs telemetry websocket at boot.
#
# Usage:
#   ./controller/scripts/jetson/helper/install_telemetry_service.sh --token <shared-token>
#   ./controller/scripts/jetson/helper/install_telemetry_service.sh --token <shared-token> --no-start
#   ./controller/scripts/jetson/helper/install_telemetry_service.sh --token <shared-token> --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
START_SCRIPT="$SCRIPT_DIR/start_telemetry_websocket.sh"

SERVICE_NAME="krabby-telemetry"
ENV_FILE="/etc/default/${SERVICE_NAME}"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

ENABLE_NOW=1
DRY_RUN=0

TOKEN="${KRABBY_TELEMETRY_TOKEN:-}"
IMAGE="${KRABBY_LOCOMOTION_IMAGE:-krabby-locomotion:latest}"
MCU_PORT="${KRABBY_MCU_PORT:-/dev/ttyACM0}"
WS_HOST="${KRABBY_TELEMETRY_WS_HOST:-0.0.0.0}"
WS_PORT="${KRABBY_TELEMETRY_WS_PORT:-8787}"
WS_PATH="${KRABBY_TELEMETRY_WS_PATH:-/krabby/telemetry}"
WS_HZ="${KRABBY_TELEMETRY_WS_HZ:-10}"
FAKE_DATA="${KRABBY_TELEMETRY_FAKE_DATA:-0}"
USE_NVIDIA_RUNTIME="${KRABBY_USE_NVIDIA_RUNTIME:-0}"
DOCKER_NETWORK_MODE="${KRABBY_DOCKER_NETWORK_MODE:-auto}"

usage() {
  cat <<'EOF'
Usage: ./controller/scripts/jetson/helper/install_telemetry_service.sh --token <shared-token> [options]

Options:
  --token <value>                Required if KRABBY_TELEMETRY_TOKEN is not already set.
  --service-name <value>         Service name (default: krabby-telemetry).
  --no-start                     Install/enable at boot, but do not start immediately.
  --dry-run                      Print actions without writing files or running systemctl.
  -h, --help                     Show this help.

The script writes:
  - /etc/default/<service-name>
  - /etc/systemd/system/<service-name>.service
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --token)
      TOKEN="${2:-}"
      shift 2
      ;;
    --service-name)
      SERVICE_NAME="${2:-}"
      shift 2
      ;;
    --no-start)
      ENABLE_NOW=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$TOKEN" ]]; then
  echo "Error: telemetry token is required. Pass --token or set KRABBY_TELEMETRY_TOKEN." >&2
  exit 1
fi

if [[ ! -x "$START_SCRIPT" ]]; then
  echo "Error: start script not found or not executable: $START_SCRIPT" >&2
  exit 1
fi

if [[ "$FAKE_DATA" != "0" && "$FAKE_DATA" != "1" ]]; then
  echo "Error: KRABBY_TELEMETRY_FAKE_DATA must be 0 or 1 (got: $FAKE_DATA)" >&2
  exit 1
fi

if ! command -v systemctl >/dev/null 2>&1; then
  if [[ "$DRY_RUN" == "0" ]]; then
    echo "Error: systemctl not found. This installer requires systemd." >&2
    exit 1
  fi
  echo "Warning: systemctl not found; continuing because --dry-run was requested." >&2
fi

if [[ -z "$SERVICE_NAME" ]]; then
  echo "Error: service name cannot be empty." >&2
  exit 1
fi

ENV_FILE="/etc/default/${SERVICE_NAME}"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

if [[ $EUID -eq 0 ]]; then
  SUDO=()
else
  SUDO=(sudo)
fi

run_cmd() {
  printf "+ "
  printf "%q " "$@"
  printf "\n"
  if [[ "$DRY_RUN" == "0" ]]; then
    "$@"
  fi
}

tmp_env="$(mktemp)"
tmp_service="$(mktemp)"
cleanup() {
  rm -f "$tmp_env" "$tmp_service"
}
trap cleanup EXIT

cat >"$tmp_env" <<EOF
KRABBY_TELEMETRY_TOKEN=$TOKEN
KRABBY_LOCOMOTION_IMAGE=$IMAGE
KRABBY_MCU_PORT=$MCU_PORT
KRABBY_TELEMETRY_WS_HOST=$WS_HOST
KRABBY_TELEMETRY_WS_PORT=$WS_PORT
KRABBY_TELEMETRY_WS_PATH=$WS_PATH
KRABBY_TELEMETRY_WS_HZ=$WS_HZ
KRABBY_TELEMETRY_FAKE_DATA=$FAKE_DATA
KRABBY_USE_NVIDIA_RUNTIME=$USE_NVIDIA_RUNTIME
KRABBY_DOCKER_NETWORK_MODE=$DOCKER_NETWORK_MODE
EOF

cat >"$tmp_service" <<EOF
[Unit]
Description=Krabby Telemetry WebSocket
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service

[Service]
Type=simple
EnvironmentFile=$ENV_FILE
WorkingDirectory=$REPO_ROOT
ExecStart=$START_SCRIPT
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

run_cmd "${SUDO[@]}" install -m 0644 "$tmp_env" "$ENV_FILE"
run_cmd "${SUDO[@]}" install -m 0644 "$tmp_service" "$SERVICE_FILE"
run_cmd "${SUDO[@]}" systemctl daemon-reload
if [[ "$ENABLE_NOW" == "1" ]]; then
  run_cmd "${SUDO[@]}" systemctl enable --now "${SERVICE_NAME}.service"
else
  run_cmd "${SUDO[@]}" systemctl enable "${SERVICE_NAME}.service"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run complete. No files were written."
  echo "Would install: ${SERVICE_NAME}.service"
  echo "Would write env file: $ENV_FILE"
else
  echo "Installed ${SERVICE_NAME}.service"
  echo "Environment file: $ENV_FILE"
  if [[ "$ENABLE_NOW" == "1" ]]; then
    echo "Service started. Check logs with:"
    echo "  journalctl -u ${SERVICE_NAME}.service -f"
  else
    echo "Service enabled for boot. Start now with:"
    echo "  sudo systemctl start ${SERVICE_NAME}.service"
  fi
fi
