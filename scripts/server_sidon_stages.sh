#!/usr/bin/env bash
# After server_clean_stages.sh: Sidon speech restoration of every separated training clip (ttsdata.restoration), then
# both manifests rebuilt on wavs_sidon. Per root: cut each clip from the episode with 0.5 s context (cut_context.py),
# remove music from the cuts (separate_vocals.py, all separated), Sidon + trim back + ending check (sidon_restore.py),
# word check with Whisper (sidon_wordcheck.py), re-audit silence/loudness on wavs_sidon. The previous manifests move to data/voxcpm_sep and data/voxcpm_stage2_sep.
# Resumable: each stage writes data/logs/sidon_<stage>_done and is skipped when it exists.
#   tmux new -d -s sidon 'scripts/server_sidon_stages.sh >> data/logs/sidon_stages.log 2>&1'
# Env: PY (project venv), SEP_PY (audio-separator venv), SIDON_PY (torch + transformers venv), WORKERS, SEP_JOBS,
#      SEP_BATCH, SIDON_JOBS, MAX_HOURS, AUTO_CONF, MIN_TAG, MANIFEST_EXTRA (extra make_voxcpm_manifest.py flags).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
SEP_PY="${SEP_PY:-$HOME/tts/venv-sep/bin/python}"
SIDON_PY="${SIDON_PY:-$HOME/tts/venv-voxcpm/bin/python}"
WORKERS="${WORKERS:-8}"
SEP_JOBS="${SEP_JOBS:-6}"
SEP_BATCH="${SEP_BATCH:-64}"
SIDON_JOBS="${SIDON_JOBS:-2}"
MAX_HOURS="${MAX_HOURS:-60}"
AUTO_CONF="${AUTO_CONF:-angry=0.70,sad=0.85,happy=0.99}"
MIN_TAG="${MIN_TAG:-150}"
MANIFEST_EXTRA="${MANIFEST_EXTRA:-}"
LOG=data/logs
mkdir -p "$LOG"

stage() {  # stage <name> <command...>
    local name=$1; shift
    if [ -e "$LOG/sidon_${name}_done" ]; then echo "skip $name (done)"; return; fi
    echo "== $name  start $(date '+%F %T')"
    local t0=$SECONDS rc=0
    "$@" > "$LOG/sidon_${name}.log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "== $name  FAILED  $(date '+%F %T')  exit $rc, see $LOG/sidon_${name}.log"
        exit "$rc"
    fi
    touch "$LOG/sidon_${name}_done"
    echo "== $name  done  $(date '+%F %T')  $(( (SECONDS - t0) / 60 )) min"
}

parallel() {  # parallel <jobs> <log prefix> <merge command> -- <command...>: run with --shard k/N, then merge
    local n=$1 prefix=$2 merge=$3 k rc=0 pids=()
    shift 4
    for k in $(seq 0 $((n - 1))); do
        "$@" --shard "$k/$n" > "${prefix}_$k.log" 2>&1 &
        pids+=($!)
    done
    for k in "${pids[@]}"; do wait "$k" || rc=$?; done
    $merge || rc=$?  # merge what finished, even after a failure
    return "$rc"
}

separate() {  # separate <root>: music removal on the context cuts, every clip separated
    local root=$1
    parallel "$SEP_JOBS" "$LOG/sidon_sep_$(basename "$root")" \
        "$SEP_PY scripts/separate_vocals.py --root $root --quality-file ctx_quality.jsonl --merge" -- \
        "$SEP_PY" scripts/separate_vocals.py --root "$root" --wavs wavs_ctx --out wavs_ctx_clean \
        --quality-file ctx_quality.jsonl --ids-from-wavs --threshold-db -200 --batch "$SEP_BATCH"
}

restore() {  # restore <root>: Sidon on the separated context cuts
    local root=$1
    parallel "$SIDON_JOBS" "$LOG/sidon_restore_$(basename "$root")" \
        "$SIDON_PY scripts/sidon_restore.py --root $root --merge" -- \
        "$SIDON_PY" scripts/sidon_restore.py --root "$root"
}

wordcheck() {  # wordcheck <root>: Whisper on the restored clips; clips that lost words fall back
    local root=$1
    parallel "$SIDON_JOBS" "$LOG/sidon_words_$(basename "$root")" \
        "$SIDON_PY scripts/sidon_wordcheck.py --root $root --merge" -- \
        "$SIDON_PY" scripts/sidon_wordcheck.py --root "$root"
}

audit() {  # audit <root>: silence edges / loudness on wavs_sidon; exit 1 = problems listed, not a crash
    local root=$1 rc=0 t0
    t0=$(mktemp)
    "$PY" scripts/audit_quality.py --root "$root" --wavs wavs_sidon --utmos 0 --workers "$WORKERS" || rc=$?
    if [ "$rc" -eq 1 ] && [ "$root/AUDIT.md" -nt "$t0" ]; then
        echo "problems listed in $root/AUDIT.md; counted as done"
        rc=0
    fi
    rm -f "$t0"
    return "$rc"
}

keep_previous() {  # move the wavs_clean manifests aside once (same depth under data/, relative paths stay valid)
    for d in voxcpm voxcpm_stage2; do
        if [ -d "data/${d}_sep" ]; then echo "data/${d}_sep exists"; continue; fi
        mv "data/$d" "data/${d}_sep"
        echo "data/$d -> data/${d}_sep"
    done
}

for r in dataset_events dataset_long dataset; do
    stage "cut_$r"     "$PY" scripts/cut_context.py --root "data/$r" --workers "$WORKERS"
    stage "sep_$r"     separate "data/$r"
    stage "restore_$r" restore "data/$r"
    stage "words_$r"   wordcheck "data/$r"
    stage "audit_$r"   audit "data/$r"
done

SIDON="--wavs wavs_sidon --long-wavs wavs_sidon --events-wavs wavs_sidon --max-hours $MAX_HOURS"
stage keep_previous keep_previous
stage manifest1 "$PY" scripts/make_voxcpm_manifest.py --stage 1 --out data/voxcpm $SIDON $MANIFEST_EXTRA
stage manifest2 "$PY" scripts/make_voxcpm_manifest.py --stage 2 --out data/voxcpm_stage2 $SIDON $MANIFEST_EXTRA \
    --auto-styles --auto-conf "$AUTO_CONF" --min-tag "$MIN_TAG"
touch "$LOG/sidon_stages_done"
echo "all sidon stages done $(date '+%F %T')"
