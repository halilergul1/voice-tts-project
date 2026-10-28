#!/usr/bin/env bash
# Score every stage 1 LoRA checkpoint with scripts/eval_tts.py: the 50 corpus held-out sentences and the first 100
# Freya-TR-Eval sentences (rows s1_step<N>_corpus / s1_step<N>_freya100 in data/eval/summary.md). Resumable: a scored
# checkpoint writes data/logs/score_s1_step<N>_done. Run after training (both on one 32 GB GPU do not fit).
#   tmux new -d -s score1 'scripts/score_stage1.sh >> data/logs/score_stage1.log 2>&1'
# Env: PY, MODELS, CKPT (checkpoint folder), FREYA_N.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/tts/venv-voxcpm/bin/python}"
MODELS="${MODELS:-$HOME/tts/models}"
CKPT="${CKPT:-$HOME/tts/checkpoints/stage1}"
FREYA_N="${FREYA_N:-100}"
LOG=data/logs
for d in "$CKPT"/step_*; do
    [ -f "$d/lora_weights.safetensors" ] || continue
    n=$((10#${d##*step_}))
    [ "$n" -eq 0 ] && continue  # step 0 is the untrained adapter
    tag="s1_step$n"
    if [ -e "$LOG/score_${tag}_done" ]; then echo "skip $tag (done)"; continue; fi
    echo "== $tag  start $(date '+%F %T')"
    "$PY" scripts/eval_tts.py all --system voxcpm2-lora --model-dir "$MODELS/VoxCPM2" --lora "$d" --run "${tag}_corpus" \
        --sentences data/eval/corpus_heldout.txt > "$LOG/score_${tag}_corpus.log" 2>&1
    "$PY" scripts/eval_tts.py all --system voxcpm2-lora --model-dir "$MODELS/VoxCPM2" --lora "$d" --run "${tag}_freya$FREYA_N" \
        --sentences data/eval/freya_tr_eval.txt --limit "$FREYA_N" > "$LOG/score_${tag}_freya.log" 2>&1
    touch "$LOG/score_${tag}_done"
    echo "== $tag  done  $(date '+%F %T')"
done
touch "$LOG/score_stage1_done"
echo "scoring done $(date '+%F %T')"
