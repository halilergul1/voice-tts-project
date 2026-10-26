#!/usr/bin/env bash
# Whole-catalogue driver for a Linux GPU box. Runs detached (tmux):
#   - N_DL downloaders walk the metadata csv (sharded, resumable)
#   - N_SEG segmenters (lock-protected) keep cutting newly downloaded episodes
#   - N_WORKERS Whisper workers keep verifying newly segmented episodes
#   - when everything is drained: build + integrity audit, then data/logs/pipeline_done
# Re-running after an interruption resumes everything; nothing is redone.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.deno/bin:$PATH" PYTHONUNBUFFERED=1 TTSDATA_TORCH_THREADS="${TTSDATA_TORCH_THREADS:-2}"
CSV="${CSV:-meta_datasets/engelsiz_TRT_videos_metadata.csv}"
N_DL="${N_DL:-2}"; N_SEG="${N_SEG:-4}"; N_WORKERS="${N_WORKERS:-4}"
SERIES="${SERIES:-}"; LIMIT="${LIMIT:-}"
PY=.venv/bin/ttsdata
mkdir -p data/logs
rm -f data/logs/pipeline_done data/logs/downloads_done data/logs/segments_done
log() { echo "$(date '+%F %T') $*" | tee -a data/logs/pipeline.log; }

# ---------------------------------------------------------------- downloaders
log "start: downloaders=$N_DL segmenters=$N_SEG whisper_workers=$N_WORKERS csv=$CSV ${SERIES:+series=$SERIES} ${LIMIT:+limit=$LIMIT}"
(
  pids=()
  for ((k = 0; k < N_DL; k++)); do
    $PY download-trt --csv "$CSV" --shard "$k/$N_DL" ${SERIES:+--series "$SERIES"} ${LIMIT:+--limit "$LIMIT"} \
       >> "data/logs/download_$k.log" 2>&1 &
    pids+=($!); sleep 5
  done
  wait "${pids[@]}"
  touch data/logs/downloads_done
) &

# ---------------------------------------------------------------- segmenters
(
  while :; do
    pids=()
    for ((k = 0; k < N_SEG; k++)); do
      $PY segment >> "data/logs/segment_$k.log" 2>&1 & pids+=($!); sleep 2
    done
    wait "${pids[@]}"
    if [ -f data/logs/downloads_done ]; then
      # one more pass to catch the last downloads, then declare done
      $PY segment >> data/logs/segment_0.log 2>&1
      touch data/logs/segments_done; break
    fi
    sleep 30
  done
) &

# ---------------------------------------------------------------- whisper workers
while :; do
  scripts/verify_parallel.sh "$N_WORKERS" >> data/logs/verify_passes.log 2>&1
  log "verify pass: raw=$(ls data/raw/*/ 2>/dev/null | grep -vc '^$' ) segmented=$(ls -d data/segments/*/*/ 2>/dev/null | wc -l) clips_verified=$(cat data/segments/*/*/asr.jsonl 2>/dev/null | wc -l) disk_free=$(df -h . | awk 'NR==2{print $4}')"
  if [ -f data/logs/segments_done ]; then
    scripts/verify_parallel.sh "$N_WORKERS" >> data/logs/verify_passes.log 2>&1
    break
  fi
  sleep 60
done

log "all verified; building"
$PY build > data/logs/build.log 2>&1 && log "build ok" || log "build FAILED"
.venv/bin/python scripts/audit_dataset.py --skip-asr > data/logs/audit_quick.log 2>&1
tail -3 data/logs/audit_quick.log | tee -a data/logs/pipeline.log
touch data/logs/pipeline_done
log "pipeline_done"
