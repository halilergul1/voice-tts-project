#!/usr/bin/env bash
# Everything after `ttsdata speakers` on a finished build, in order, resumable: each stage writes
# data/logs/label_<stage>_done and is skipped on the next run when that marker exists (delete it to redo a stage).
# Run detached on the server:  nohup setsid scripts/server_label_stages.sh > data/logs/label_stages.log 2>&1 &
# Env: PY (python), WORKERS (quality audit processes), MAX_HOURS (manifest cap), AUTO_CONF (calibrated cut-offs).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
WORKERS="${WORKERS:-12}"
MAX_HOURS="${MAX_HOURS:-60}"
AUTO_CONF="${AUTO_CONF:-angry=0.70,sad=0.85,happy=0.99}"
LOG=data/logs
mkdir -p "$LOG"

stage() {  # stage <name> <command...>
    local name=$1; shift
    if [ -e "$LOG/label_${name}_done" ]; then echo "skip $name (done)"; return; fi
    echo "== $name  start $(date '+%F %T')"
    local t0=$SECONDS
    "$@" > "$LOG/label_${name}.log" 2>&1
    touch "$LOG/label_${name}_done"
    echo "== $name  done  $(date '+%F %T')  $(( (SECONDS - t0) / 60 )) min"
}

stage quality        "$PY" scripts/audit_quality.py --root data/dataset --utmos 0 --workers "$WORKERS"
stage merge          "$PY" -m ttsdata merge-clips
stage quality_long   "$PY" scripts/audit_quality.py --root data/dataset_long --utmos 0 --workers "$WORKERS"
stage soundtags      "$PY" -m ttsdata soundtags
stage events         "$PY" -m ttsdata extend-events
stage verify_events  "$PY" -m ttsdata verify-events
stage quality_events "$PY" scripts/audit_quality.py --root data/dataset_events --utmos 0 --workers "$WORKERS"
stage emotions       "$PY" -m ttsdata emotions

FETCH=""; [ -d data/fleurs_tr ] && [ -n "$(ls data/fleurs_tr/*.wav 2>/dev/null | head -1)" ] || FETCH="--fetch-fleurs"
stage manifest1 "$PY" scripts/make_voxcpm_manifest.py --stage 1 --out data/voxcpm --wavs wavs --long-wavs wavs --events-wavs wavs \
    --max-hours "$MAX_HOURS" $FETCH
stage manifest2 "$PY" scripts/make_voxcpm_manifest.py --stage 2 --out data/voxcpm_stage2 --wavs wavs --long-wavs wavs --events-wavs wavs \
    --max-hours "$MAX_HOURS" --auto-styles --auto-conf "$AUTO_CONF"
touch "$LOG/label_stages_done"
echo "all stages done $(date '+%F %T')"
