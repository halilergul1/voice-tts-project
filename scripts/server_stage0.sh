#!/usr/bin/env bash
# Stage 0 baselines on astrobot: zero-shot VoxCPM2 (voice-design prompt) and Trendyol-TTS on Freya-TR-Eval and the
# corpus held-out sentences, scored by scripts/eval_tts.py (rows land in data/eval/summary.md).
# Resumable: each run writes data/logs/stage0_<run>_done and is skipped when it exists.
#   tmux new -d -s stage0 'scripts/server_stage0.sh >> data/logs/stage0.log 2>&1'
# Needs scripts/astrobot_setup_tts.sh first. Env: PY (venv with voxcpm), MODELS (snapshot folder).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/tts/venv-voxcpm/bin/python}"
MODELS="${MODELS:-$HOME/tts/models}"
LOG=data/logs
mkdir -p "$LOG"

run() {  # run <run name> <system> <model dir> <sentences>
    local name=$1 system=$2 model=$3 sentences=$4
    if [ -e "$LOG/stage0_${name}_done" ]; then echo "skip $name (done)"; return; fi
    echo "== $name  start $(date '+%F %T')"
    local rc=0
    "$PY" scripts/eval_tts.py all --system "$system" --model-dir "$MODELS/$model" --run "$name" \
        --sentences "$sentences" > "$LOG/stage0_${name}.log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then echo "== $name  FAILED  $(date '+%F %T')  exit $rc, see $LOG/stage0_${name}.log"; exit "$rc"; fi
    touch "$LOG/stage0_${name}_done"
    echo "== $name  done  $(date '+%F %T')"
}

[ -s data/eval/freya_tr_eval.txt ] || "$PY" scripts/eval_tts.py fetch
run s0_voxcpm2_zeroshot voxcpm2  VoxCPM2      data/eval/freya_tr_eval.txt
run s0_trendyol         trendyol Trendyol-TTS data/eval/freya_tr_eval.txt
run s0_voxcpm2_corpus   voxcpm2  VoxCPM2      data/eval/corpus_heldout.txt
run s0_trendyol_corpus  trendyol Trendyol-TTS data/eval/corpus_heldout.txt
touch "$LOG/stage0_done"
echo "stage 0 done $(date '+%F %T')"
