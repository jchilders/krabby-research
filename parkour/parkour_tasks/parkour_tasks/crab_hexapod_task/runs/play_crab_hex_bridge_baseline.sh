#!/usr/bin/env bash
# Play bundled teacher-bridge baseline (Appendix D).
# Usage: play_crab_hex_bridge_baseline.sh <path-to-usda> <path-to-model.pt>
set -euo pipefail

KRABBY_ROOT="${KRABBY_ROOT:-/home/sanjay/Projects/krabby}"
RUNS_DIR="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hexapod_task/runs"

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <path-to-usda> <path-to-model.pt>" >&2
  exit 2
fi

USD_PATH="$1"
CHECKPOINT_PATH="$2"

if [[ ! -f "$USD_PATH" ]]; then
  echo "USD file not found: $USD_PATH" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint file not found: $CHECKPOINT_PATH" >&2
  exit 1
fi

export KRABBY_HEX_USD_PATH="$USD_PATH"
export KRABBY_HEX_SPAWN_Z="${KRABBY_HEX_SPAWN_Z:-1.05}"
export KRABBY_HEX_TRAIN_EASY=1
export PYTHONPATH="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks:$KRABBY_ROOT/krabby-research/parkour:${PYTHONPATH:-}"

cd "$KRABBY_ROOT/krabby-research/parkour"

echo "Playing crab hex teacher bridge baseline"
echo "  USD:        $KRABBY_HEX_USD_PATH"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  TRAIN_EASY: $KRABBY_HEX_TRAIN_EASY"
echo

"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p "$KRABBY_ROOT/krabby-research/parkour/scripts/rsl_rl/play.py" \
  --task Isaac-Crab-Hex-Teacher-Play-v0 \
  --num_envs 1 \
  --real-time \
  --checkpoint "$CHECKPOINT_PATH"
