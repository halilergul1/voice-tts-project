#!/usr/bin/env bash
# Clean and Sidon-restore more of the stage 1 strict tier, for a bigger clean-only training set (stage 1 run 5b).
# The 60 h runs only cleaned the clips of their 60 h slices; this picks a bigger raw slice of the same tier (same seed,
# so it holds the old slice), runs the same steps as server_clean_stages.sh + server_sidon_stages.sh on the clips not
# done yet (every step skips finished clips), then builds the restored-only manifest. Existing manifests are untouched.
#   tmux new -d -s moreclean 'scripts/server_more_clean.sh >> data/logs/moreclean.log 2>&1'
# Env: RAW_HOURS (raw slice size, default 160), OUT_HOURS (final manifest cap, default 150), OUT (default
#      data/voxcpm_v5b), PY, SEP_PY, SIDON_PY, WORKERS, SEP_JOBS, SEP_BATCH, SIDON_JOBS.
# Resumable: each stage writes data/logs/more_<stage>_done.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
SEP_PY="${SEP_PY:-$HOME/tts/venv-sep/bin/python}"
SIDON_PY="${SIDON_PY:-$HOME/tts/venv-voxcpm/bin/python}"
WORKERS="${WORKERS:-8}"
SEP_JOBS="${SEP_JOBS:-6}"
SEP_BATCH="${SEP_BATCH:-64}"
SIDON_JOBS="${SIDON_JOBS:-2}"
RAW_HOURS="${RAW_HOURS:-160}"
OUT_HOURS="${OUT_HOURS:-150}"
OUT="${OUT:-data/voxcpm_v5b}"
RAWSLICE="data/voxcpm_rawslice_${RAW_HOURS}h"
LOG=data/logs
mkdir -p "$LOG"

stage() {  # stage <name> <command...>
    local name=$1; shift
    if [ -e "$LOG/more_${name}_done" ]; then echo "skip $name (done)"; return; fi
    echo "== $name  start $(date '+%F %T')"
    local t0=$SECONDS rc=0
    "$@" > "$LOG/more_${name}.log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "== $name  FAILED  $(date '+%F %T')  exit $rc, see $LOG/more_${name}.log"
        exit "$rc"
    fi
    touch "$LOG/more_${name}_done"
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

audit() {  # audit <root> <wavs>: exit 1 = problems listed in AUDIT.md, not a crash
    local root=$1 wavs=$2 rc=0 t0
    t0=$(mktemp)
    "$PY" scripts/audit_quality.py --root "$root" --wavs "$wavs" --utmos 0 --workers "$WORKERS" || rc=$?
    if [ "$rc" -eq 1 ] && [ "$root/AUDIT.md" -nt "$t0" ]; then
        echo "problems listed in $root/AUDIT.md; counted as done"
        rc=0
    fi
    rm -f "$t0"
    return "$rc"
}

separate_clips() {  # separate_clips <root>: music removal of the raw slice's clips, every clip separated
    local root=$1
    parallel "$SEP_JOBS" "$LOG/more_sep_$(basename "$root")" "$SEP_PY scripts/separate_vocals.py --root $root --merge" -- \
        "$SEP_PY" scripts/separate_vocals.py --root "$root" --manifests "$RAWSLICE/train.jsonl,$RAWSLICE/val.jsonl" \
        --batch "$SEP_BATCH" --threshold-db -200
}

separate_ctx() {  # separate_ctx <root>: music removal on the context cuts (as server_sidon_stages.sh)
    local root=$1
    parallel "$SEP_JOBS" "$LOG/more_sepctx_$(basename "$root")" \
        "$SEP_PY scripts/separate_vocals.py --root $root --quality-file ctx_quality.jsonl --merge" -- \
        "$SEP_PY" scripts/separate_vocals.py --root "$root" --wavs wavs_ctx --out wavs_ctx_clean \
        --quality-file ctx_quality.jsonl --ids-from-wavs --threshold-db -200 --batch "$SEP_BATCH"
}

restore() {
    local root=$1
    parallel "$SIDON_JOBS" "$LOG/more_restore_$(basename "$root")" "$SIDON_PY scripts/sidon_restore.py --root $root --merge" -- \
        "$SIDON_PY" scripts/sidon_restore.py --root "$root"
}

wordcheck() {
    local root=$1
    parallel "$SIDON_JOBS" "$LOG/more_words_$(basename "$root")" "$SIDON_PY scripts/sidon_wordcheck.py --root $root --merge" -- \
        "$SIDON_PY" scripts/sidon_wordcheck.py --root "$root"
}

stage rawslice "$PY" scripts/make_voxcpm_manifest.py --stage 1 --out "$RAWSLICE" --wavs wavs --long-wavs wavs \
    --events-wavs wavs --max-hours "$RAW_HOURS"
for r in dataset_long dataset; do
    stage "sep_$r"      separate_clips "data/$r"
    stage "quality_$r"  audit "data/$r" wavs_clean
    stage "cut_$r"      "$PY" scripts/cut_context.py --root "data/$r" --workers "$WORKERS"
    stage "sepctx_$r"   separate_ctx "data/$r"
    stage "restore_$r"  restore "data/$r"
    stage "words_$r"    wordcheck "data/$r"
    stage "audit_$r"    audit "data/$r" wavs_sidon
done
stage manifest "$PY" scripts/make_voxcpm_manifest.py --stage 1 --out "$OUT" --wavs wavs_sidon --long-wavs wavs_sidon \
    --events-wavs wavs_clean --max-hours "$OUT_HOURS" --restored-only --replay-dir data/fleurs_tr_sidon \
    --speaker-gender data/dataset/speaker_gender.jsonl --voice-desc-lang en
touch "$LOG/more_clean_done"
echo "all more-clean stages done $(date '+%F %T')"
