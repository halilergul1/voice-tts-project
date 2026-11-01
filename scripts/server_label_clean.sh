#!/usr/bin/env bash
# Stage 2 Task 3A: clean and Sidon-restore every labelled clip (styles, tags), so the stage 2 build can be restored-only.
# Builds a raw-audio stage 2 manifest of labelled clips only (data/voxcpm_stage2_rawlabels: its tier_report.md lists the
# labelled hours before cleaning), then runs the steps of server_more_clean.sh on those clips: separation, audit, context
# cuts, separation of the cuts, Sidon, word check, audit for dataset_long and dataset (the dataset pass also separates
# the base clip of every event clip), separation + audit only for dataset_events. Every step skips finished clips.
# The context cuts (and so Sidon) follow the manifest too, not every separated clip: a clip an earlier, wider selection
# separated (e.g. the 1.5-3 s lines dropped in Task 3D) stops there.
#   tmux new -d -s lclean 'scripts/server_label_clean.sh >> data/logs/lclean.log 2>&1'
# Env: RAW_HOURS (default 400), RAWLABELS (default data/voxcpm_stage2_rawlabels), NARR (narration_styles.jsonl from
#      Task 3B, optional), LOG (default data/logs), PY, SEP_PY, SIDON_PY, WORKERS, SEP_JOBS, SEP_BATCH, SIDON_JOBS.
# Resumable: each stage writes $LOG/lclean_<stage>_done; all done: $LOG/lclean_done.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
SEP_PY="${SEP_PY:-$HOME/tts/venv-sep/bin/python}"
SIDON_PY="${SIDON_PY:-$HOME/tts/venv-voxcpm/bin/python}"
WORKERS="${WORKERS:-8}"
SEP_JOBS="${SEP_JOBS:-6}"
SEP_BATCH="${SEP_BATCH:-64}"
SIDON_JOBS="${SIDON_JOBS:-2}"
RAW_HOURS="${RAW_HOURS:-400}"
RAWLABELS="${RAWLABELS:-data/voxcpm_stage2_rawlabels}"
NARR="${NARR:-}"
LOG="${LOG:-data/logs}"
mkdir -p "$LOG"

stage() {  # stage <name> <command...>
    local name=$1; shift
    if [ -e "$LOG/lclean_${name}_done" ]; then echo "skip $name (done)"; return; fi
    echo "== $name  start $(date '+%F %T')"
    local t0=$SECONDS rc=0
    "$@" > "$LOG/lclean_${name}.log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "== $name  FAILED  $(date '+%F %T')  exit $rc, see $LOG/lclean_${name}.log"
        exit "$rc"
    fi
    touch "$LOG/lclean_${name}_done"
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

separate_clips() {  # separate_clips <root>: music removal of the labelled clips (and, for dataset, event bases)
    local root=$1
    parallel "$SEP_JOBS" "$LOG/lclean_sep_$(basename "$root")" "$SEP_PY scripts/separate_vocals.py --root $root --merge" -- \
        "$SEP_PY" scripts/separate_vocals.py --root "$root" --manifests "$RAWLABELS/train.jsonl,$RAWLABELS/val.jsonl" \
        --batch "$SEP_BATCH" --threshold-db -200
}

separate_ctx() {  # separate_ctx <root>: music removal on the context cuts (as server_sidon_stages.sh)
    local root=$1
    parallel "$SEP_JOBS" "$LOG/lclean_sepctx_$(basename "$root")" \
        "$SEP_PY scripts/separate_vocals.py --root $root --quality-file ctx_quality.jsonl --merge" -- \
        "$SEP_PY" scripts/separate_vocals.py --root "$root" --wavs wavs_ctx --out wavs_ctx_clean \
        --quality-file ctx_quality.jsonl --ids-from-wavs --threshold-db -200 --batch "$SEP_BATCH"
}

restore() {
    local root=$1
    parallel "$SIDON_JOBS" "$LOG/lclean_restore_$(basename "$root")" "$SIDON_PY scripts/sidon_restore.py --root $root --merge" -- \
        "$SIDON_PY" scripts/sidon_restore.py --root "$root"
}

wordcheck() {
    local root=$1
    parallel "$SIDON_JOBS" "$LOG/lclean_words_$(basename "$root")" "$SIDON_PY scripts/sidon_wordcheck.py --root $root --merge" -- \
        "$SIDON_PY" scripts/sidon_wordcheck.py --root "$root"
}

# labelled clips only (no plain rows), every label at full quota, unconfirmed laughs and breaths too, so a later yes
# to them needs no second pass; Task 3D: subtitle kept up to a 20% Whisper mismatch (short 1.5-3 s lines failed
# the listening check: emotion labels do not hold on them)
stage rawlabels "$PY" scripts/make_voxcpm_manifest.py --stage 2 --out "$RAWLABELS" --wavs wavs --long-wavs wavs \
    --events-wavs wavs --auto-styles --auto-conf angry=0.70,sad=0.85,happy=0.99 --min-tag 150 \
    --include-unconfirmed-tags cry,laugh,breath --plain-rows none --label-first --label-quota 1000000 \
    --max-cer 0.20 \
    --label-max-share 1.0 --label-repeat 1 --replay-share 0 --max-hours "$RAW_HOURS" --max-series-share 1.0 \
    ${NARR:+--narration-styles "$NARR"}
stage sep_dataset_events      separate_clips data/dataset_events
stage quality_dataset_events  audit data/dataset_events wavs_clean
for r in dataset_long dataset; do
    stage "sep_$r"      separate_clips "data/$r"
    stage "quality_$r"  audit "data/$r" wavs_clean
    stage "cut_$r"      "$PY" scripts/cut_context.py --root "data/$r" --workers "$WORKERS" \
        --manifests "$RAWLABELS/train.jsonl,$RAWLABELS/val.jsonl"
    stage "sepctx_$r"   separate_ctx "data/$r"
    stage "restore_$r"  restore "data/$r"
    stage "words_$r"    wordcheck "data/$r"
    stage "audit_$r"    audit "data/$r" wavs_sidon
done
touch "$LOG/lclean_done"
echo "all label-clean stages done $(date '+%F %T')"
