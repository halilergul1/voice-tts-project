#!/usr/bin/env bash
# Stage 2 run 2a on astrobot (training/voxcpm/stage2_style_lm.yaml): the stage 1 launcher with stage 2 defaults.
# Resumes from the latest checkpoint when restarted. Writes data/logs/stage2a_train_done on success.
#   tmux new -d -s stage2a 'bash ~/voice-data/scripts/train_stage2.sh'
# Env: CONFIG, LOGFILE, DONE_MARKER (and PY, VOXCPM as in train_stage1.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CONFIG="${CONFIG:-$ROOT/training/voxcpm/stage2_style_lm.yaml}"
export LOGFILE="${LOGFILE:-$HOME/tts/logs/stage2a_train.log}"
export DONE_MARKER="${DONE_MARKER:-stage2a_train_done}"
exec bash "$ROOT/scripts/train_stage1.sh"
