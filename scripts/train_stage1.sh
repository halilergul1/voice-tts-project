#!/usr/bin/env bash
# Stage 1 prosody LoRA on astrobot (training/voxcpm/stage1_prosody_lora.yaml). Resumes from the latest checkpoint
# when restarted. Writes data/logs/stage1_train_done on success.
#   tmux new -d -s stage1 'scripts/train_stage1.sh'
# Env: PY (venv with voxcpm), VOXCPM (upstream repo with the trainer), CONFIG.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
PY="${PY:-$HOME/tts/venv-voxcpm/bin/python}"
VOXCPM="${VOXCPM:-$HOME/tts/VoxCPM}"
CONFIG="${CONFIG:-$ROOT/training/voxcpm/stage1_prosody_lora.yaml}"
LOGFILE="${LOGFILE:-$HOME/tts/logs/stage1_train.log}"
mkdir -p "$(dirname "$LOGFILE")" "$ROOT/data/logs"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"  # less fragmentation near the 32 GB limit
cd "$VOXCPM"
echo "== train start $(date '+%F %T') config $CONFIG" >> "$LOGFILE"
rc=0
"$PY" scripts/train_voxcpm_finetune.py --config_path "$CONFIG" >> "$LOGFILE" 2>&1 || rc=$?
echo "== train exit $rc $(date '+%F %T')" >> "$LOGFILE"
[ "$rc" -eq 0 ] && touch "$ROOT/data/logs/stage1_train_done"
exit "$rc"
