#!/usr/bin/env bash
# Stop/disable/remove the telemetry websocket systemd service.
#
# Usage:
#   ./controller/scripts/jetson/helper/uninstall_telemetry_service.sh
#   ./controller/scripts/jetson/helper/uninstall_telemetry_service.sh --keep-env
#   ./controller/scripts/jetson/helper/uninstall_telemetry_service.sh --dry-run

set -euo pipefail

SERVICE_NAME="krabby-telemetry"
KEEP_ENV=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: ./controller/scripts/jetson/helper/uninstall_telemetry_service.sh [options]

Options:
  --service-name <value>         Service name (default: krabby-telemetry).
  --keep-env                     Keep /etc/default/<service-name>.
  --dry-run                      Print actions without changing system state.
  -h, --help                     Show this help.

By default, this script:
  1) Disables/stops <service-name>.service
  2) Removes /etc/systemd/system/<service-name>.service
  3) Removes /etc/default/<service-name>
  4) Runs systemctl daemon-reload
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --service-name)
      SERVICE_NAME="${2:-}"
      shift 2
      ;;
    --keep-env)
      KEEP_ENV=1
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

if [[ -z "$SERVICE_NAME" ]]; then
  echo "Error: service name cannot be empty." >&2
  exit 1
fi

SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
ENV_FILE="/etc/default/${SERVICE_NAME}"
UNIT="${SERVICE_NAME}.service"

if [[ $EUID -eq 0 ]]; then
  SUDO=()
else
  SUDO=(sudo)
fi

if ! command -v systemctl >/dev/null 2>&1; then
  if [[ "$DRY_RUN" == "0" ]]; then
    echo "Error: systemctl not found. This uninstaller requires systemd." >&2
    exit 1
  fi
  echo "Warning: systemctl not found; continuing because --dry-run was requested." >&2
fi

run_cmd() {
  printf "+ "
  printf "%q " "$@"
  printf "\n"
  if [[ "$DRY_RUN" == "0" ]]; then
    "$@"
  fi
}

if [[ "$DRY_RUN" == "1" ]]; then
  run_cmd "${SUDO[@]}" systemctl disable --now "$UNIT"
else
  if command -v systemctl >/dev/null 2>&1; then
    if systemctl list-unit-files --type=service 2>/dev/null | grep -Eq "^${SERVICE_NAME}\\.service\\b"; then
      run_cmd "${SUDO[@]}" systemctl disable --now "$UNIT"
    else
      echo "Service not registered in systemd: $UNIT"
    fi
  fi
fi

if [[ -f "$SERVICE_FILE" ]]; then
  run_cmd "${SUDO[@]}" rm -f "$SERVICE_FILE"
else
  echo "Service file not found: $SERVICE_FILE"
fi

if [[ "$KEEP_ENV" == "0" ]]; then
  if [[ -f "$ENV_FILE" ]]; then
    run_cmd "${SUDO[@]}" rm -f "$ENV_FILE"
  else
    echo "Environment file not found: $ENV_FILE"
  fi
else
  echo "Keeping environment file: $ENV_FILE"
fi

if command -v systemctl >/dev/null 2>&1; then
  run_cmd "${SUDO[@]}" systemctl daemon-reload
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run complete. No files were removed."
else
  echo "Uninstalled $UNIT"
fi
