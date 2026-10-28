#!/usr/bin/env bash
# After server_label_stages.sh: strip background music from the clips the two training manifests use (vocal separation,
# per-clip choice of separated vocals or the original), re-measure those clips, and rebuild both manifests on wavs_clean.
# The raw-wav manifests move to data/voxcpm_rawwavs and data/voxcpm_stage2_rawwavs; their clip lists pick what to clean.
# Resumable: each stage writes data/logs/clean_<stage>_done and is skipped when it exists (delete it to redo a stage).
#   tmux new -d -s clean 'scripts/server_clean_stages.sh >> data/logs/clean_stages.log 2>&1'
# Env: PY (project venv), SEP_PY (venv with audio-separator: scripts/astrobot_setup_sep.sh), WORKERS, MAX_HOURS,
#      AUTO_CONF, MIN_TAG (as in server_label_stages.sh), SEP_BATCH (clips per separator call).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
SEP_PY="${SEP_PY:-$HOME/tts/venv-sep/bin/python}"
WORKERS="${WORKERS:-12}"
MAX_HOURS="${MAX_HOURS:-60}"
AUTO_CONF="${AUTO_CONF:-angry=0.70,sad=0.85,happy=0.99}"
MIN_TAG="${MIN_TAG:-200}"
SEP_BATCH="${SEP_BATCH:-256}"
LOG=data/logs
mkdir -p "$LOG"
RAW="data/voxcpm_rawwavs data/voxcpm_stage2_rawwavs"
MANIFESTS=$(for d in $RAW; do printf '%s/train.jsonl,%s/val.jsonl,' "$d" "$d"; done)
MANIFESTS=${MANIFESTS%,}

stage() {  # stage <name> <command...>
    local name=$1; shift
    if [ -e "$LOG/clean_${name}_done" ]; then echo "skip $name (done)"; return; fi
    echo "== $name  start $(date '+%F %T')"
    local t0=$SECONDS
    local rc=0
    "$@" > "$LOG/clean_${name}.log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "== $name  FAILED  $(date '+%F %T')  exit $rc, see $LOG/clean_${name}.log"
        exit "$rc"
    fi
    touch "$LOG/clean_${name}_done"
    echo "== $name  done  $(date '+%F %T')  $(( (SECONDS - t0) / 60 )) min"
}

audit() {  # audit <root>: as in server_label_stages.sh, measured on wavs_clean; exit 1 = problems listed, not a crash
    local root=$1 rc=0 t0
    t0=$(mktemp)
    "$PY" scripts/audit_quality.py --root "$root" --wavs wavs_clean --utmos 0 --workers "$WORKERS" || rc=$?
    if [ "$rc" -eq 1 ] && [ "$root/AUDIT.md" -nt "$t0" ]; then
        echo "problems listed in $root/AUDIT.md; counted as done"
        rc=0
    fi
    rm -f "$t0"
    return "$rc"
}

keep_raw() {  # move the raw-wav manifests aside once (same depth under data/, so their relative paths stay valid)
    for d in voxcpm voxcpm_stage2; do
        if [ -d "data/${d}_rawwavs" ]; then echo "data/${d}_rawwavs exists"; continue; fi
        mv "data/$d" "data/${d}_rawwavs"
        echo "data/$d -> data/${d}_rawwavs"
    done
}

stage keep_raw      keep_raw
stage sep_events    "$SEP_PY" scripts/separate_vocals.py --root data/dataset_events --manifests "$MANIFESTS" --batch "$SEP_BATCH"
stage sep_long      "$SEP_PY" scripts/separate_vocals.py --root data/dataset_long --manifests "$MANIFESTS" --batch "$SEP_BATCH"
stage sep_dataset   "$SEP_PY" scripts/separate_vocals.py --root data/dataset --manifests "$MANIFESTS" --batch "$SEP_BATCH"
stage quality_events audit data/dataset_events
stage quality_long   audit data/dataset_long
stage quality        audit data/dataset

CLEAN="--wavs wavs_clean --long-wavs wavs_clean --events-wavs wavs_clean --max-hours $MAX_HOURS"
stage manifest1 "$PY" scripts/make_voxcpm_manifest.py --stage 1 --out data/voxcpm $CLEAN
stage manifest2 "$PY" scripts/make_voxcpm_manifest.py --stage 2 --out data/voxcpm_stage2 $CLEAN \
    --auto-styles --auto-conf "$AUTO_CONF" --min-tag "$MIN_TAG"
touch "$LOG/clean_stages_done"
echo "all clean stages done $(date '+%F %T')"
