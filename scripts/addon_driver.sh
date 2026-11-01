#!/usr/bin/env bash
# Process extra episodes into a separate data root, never touching data/dataset (later stages keep cleaned
# audio, labels and restorations there, and `ttsdata build` refuses to wipe them).
#   1. segment whatever is in $ROOT/raw (CPU only; raw audio + subtitles copied or downloaded there first)
#   2. wait until the GPU is free: none of $WAIT_SESSIONS exist in tmux and no CUDA process for $IDLE_MIN minutes
#   3. Whisper verify, build $ROOT/dataset, speaker ids matched to the speakers of $REF, integrity audit
# Marker $ROOT/logs/addon_done. Resumable: re-running skips finished episodes.
#   tmux new-session -d -s addon 'ROOT=data_retry scripts/addon_driver.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.deno/bin:$PATH" PYTHONUNBUFFERED=1 TTSDATA_TORCH_THREADS="${TTSDATA_TORCH_THREADS:-2}"
ROOT="${ROOT:-data_retry}"; REF="${REF:-data}"
WAIT_SESSIONS="${WAIT_SESSIONS:-run5b r7 lclean lcleanresume}"; IDLE_MIN="${IDLE_MIN:-10}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
PY=".venv/bin/ttsdata --root $ROOT"
LOG="$ROOT/logs"; mkdir -p "$LOG"; rm -f "$LOG/addon_done"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG/addon.log"; }

log "start: root=$ROOT reference=$REF episodes=$(find "$ROOT/raw" -name info.json 2>/dev/null | wc -l)"
nice -n 10 $PY segment >> "$LOG/segment.log" 2>&1 && log "segment ok" || { log "segment FAILED"; exit 1; }

gpu_busy() {
  local s
  for s in $WAIT_SESSIONS; do tmux has-session -t "$s" 2>/dev/null && return 0; done
  [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]
}
log "waiting for a free GPU (tmux: $WAIT_SESSIONS; then $IDLE_MIN idle minutes)"
idle=0
while [ "$idle" -lt "$IDLE_MIN" ]; do
  if gpu_busy; then idle=0; else idle=$((idle + 1)); fi
  sleep 60
done
free_gb=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then log "only ${free_gb} GB free (< $MIN_FREE_GB): stopping"; exit 1; fi
log "GPU free, ${free_gb} GB disk free"

$PY verify >> "$LOG/verify.log" 2>&1 && log "verify ok" || { log "verify FAILED"; exit 1; }
$PY build > "$LOG/build.log" 2>&1 && log "build ok" || { log "build FAILED"; exit 1; }
$PY speakers --reference "$REF" > "$LOG/speakers.log" 2>&1 && log "speakers ok" || { log "speakers FAILED"; exit 1; }
.venv/bin/python scripts/audit_dataset.py --root "$ROOT/dataset" --skip-asr > "$LOG/audit.log" 2>&1 \
  && log "audit ok" || log "audit FAILED"
log "$(.venv/bin/python -c "import json;s=json.load(open('$ROOT/dataset/stats.json'));print('utterances', s.get('utterances'), 'hours', s.get('hours'))")"
touch "$LOG/addon_done"
log "done"
