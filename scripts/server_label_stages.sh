#!/usr/bin/env bash
# Everything after `ttsdata speakers` on a finished build, in order, resumable: each stage writes
# data/logs/label_<stage>_done and is skipped on the next run when that marker exists (delete it to redo a stage).
# Run detached on the server (tmux survives the ssh session, nohup over ssh did not):
#   tmux new -d -s label 'scripts/server_label_stages.sh >> data/logs/label_stages.log 2>&1'
# A failing stage prints "== <name>  FAILED" to that log and stops the run; fix it and start the same command again.
# Env: PY (python), WORKERS (quality audit processes), MAX_HOURS (manifest cap), AUTO_CONF (calibrated cut-offs),
#      MIN_TAG (stage 2: clips an event tag needs in the slice; 200 lets [ağlama] in, 2026-10-28).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
WORKERS="${WORKERS:-12}"
MAX_HOURS="${MAX_HOURS:-60}"
AUTO_CONF="${AUTO_CONF:-angry=0.70,sad=0.85,happy=0.99}"
MIN_TAG="${MIN_TAG:-200}"
LOG=data/logs
mkdir -p "$LOG"

stage() {  # stage <name> <command...>
    local name=$1; shift
    if [ -e "$LOG/label_${name}_done" ]; then echo "skip $name (done)"; return; fi
    echo "== $name  start $(date '+%F %T')"
    local t0=$SECONDS
    local rc=0
    "$@" > "$LOG/label_${name}.log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "== $name  FAILED  $(date '+%F %T')  exit $rc, see $LOG/label_${name}.log"
        exit "$rc"
    fi
    touch "$LOG/label_${name}_done"
    echo "== $name  done  $(date '+%F %T')  $(( (SECONDS - t0) / 60 )) min"
}

audit() {  # audit <root>: audit_quality.py exits 1 whenever AUDIT.md lists a problem (a report, not a crash)
    local root=$1 rc=0 t0
    t0=$(mktemp)
    "$PY" scripts/audit_quality.py --root "$root" --utmos 0 --workers "$WORKERS" || rc=$?
    if [ "$rc" -eq 1 ] && [ "$root/AUDIT.md" -nt "$t0" ]; then
        echo "problems listed in $root/AUDIT.md; counted as done"
        rc=0
    fi
    rm -f "$t0"
    return "$rc"
}

stage quality        audit data/dataset
stage merge          "$PY" -m ttsdata merge-clips
stage quality_long   audit data/dataset_long
stage soundtags      "$PY" -m ttsdata soundtags
stage events         "$PY" -m ttsdata extend-events
stage verify_events  "$PY" -m ttsdata verify-events
stage quality_events audit data/dataset_events
stage emotions       "$PY" -m ttsdata emotions

FETCH=""; [ -d data/fleurs_tr ] && [ -n "$(ls data/fleurs_tr/*.wav 2>/dev/null | head -1)" ] || FETCH="--fetch-fleurs"
stage manifest1 "$PY" scripts/make_voxcpm_manifest.py --stage 1 --out data/voxcpm --wavs wavs --long-wavs wavs --events-wavs wavs \
    --max-hours "$MAX_HOURS" $FETCH
stage manifest2 "$PY" scripts/make_voxcpm_manifest.py --stage 2 --out data/voxcpm_stage2 --wavs wavs --long-wavs wavs --events-wavs wavs \
    --max-hours "$MAX_HOURS" --auto-styles --auto-conf "$AUTO_CONF" --min-tag "$MIN_TAG"
touch "$LOG/label_stages_done"
echo "all stages done $(date '+%F %T')"
