#!/usr/bin/env bash
# Play stage-2b checkpoint (KRABBY_HEX_TEACHER_MODE=2b1 or 2b2).
# Bundled baselines: play_crab_hex_2b1_baseline.sh (Appendix E), play_crab_hex_2b2_baseline.sh (Appendix F).
# Usage: play_crab_hex_stage_2b.sh <2b1|2b2> <path-to-usda> <path-to-model.pt>
set -euo pipefail

KRABBY_ROOT="${KRABBY_ROOT:-/home/sanjay/Projects/krabby}"
MODE="${1:-2b1}"
USD_PATH="${2:?Usage: $0 <2b1|2b2> <usda> <checkpoint>}"
CHECKPOINT_PATH="${3:?Usage: $0 <2b1|2b2> <usda> <checkpoint>}"

if [[ "$MODE" != "2b1" && "$MODE" != "2b2" ]]; then
  echo "Mode must be 2b1 or 2b2 (got: $MODE)" >&2
  exit 2
fi
if [[ ! -f "$USD_PATH" || ! -f "$CHECKPOINT_PATH" ]]; then
  echo "USD or checkpoint not found." >&2
  exit 1
fi

export KRABBY_HEX_TEACHER_MODE="$MODE"
export KRABBY_HEX_USD_PATH="$USD_PATH"
export KRABBY_HEX_SPAWN_Z="${KRABBY_HEX_SPAWN_Z:-1.05}"
export PYTHONPATH="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks:$KRABBY_ROOT/krabby-research/parkour:${PYTHONPATH:-}"

cd "$KRABBY_ROOT/krabby-research/parkour"

echo "Playing crab hex teacher mode=$MODE"
echo "  USD:        $KRABBY_HEX_USD_PATH"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  TEACHER_MODE: $KRABBY_HEX_TEACHER_MODE"
echo

"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p "$KRABBY_ROOT/krabby-research/parkour/scripts/rsl_rl/play.py" \
  --task Isaac-Crab-Hex-Teacher-Play-v0 \
  --num_envs 1 \
  --real-time \
  --checkpoint "$CHECKPOINT_PATH"
