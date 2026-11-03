#!/usr/bin/env bash
# Stage 2 Task 6: score run 2a checkpoints. One call = one checkpoint at one LoRA strength (language model only):
#   plain   Freya 0-39 in the stage 1 setting (continue + clone clip B, no prefix): eval_tts.py all, score_styles.py
#           --target neutral (voice vs s0_voxcpm2_contB_freya40), plain_metrics.py
#   styles  style_neutral.txt (20) and style_matched_<s>.txt (10) per style in the Task 4 mode P2 (clone clip B,
#           Turkish prefix as --prompt), score_styles.py --target <s>
#   tags    tags_<tag>.txt (10) per trained tag in P2 (clone clip B, tag inline), score_styles.py --tag <tag>
# Runs are named s2a_step<N>_x<S>_<set>. Each finished call writes data/logs/score_s2a_step<N>_x<S>_done.
#   scripts/score_stage2.sh 800 1.0       # one checkpoint
#   scripts/score_stage2.sh               # every saved checkpoint that is a multiple of 400 and the last one, at 1.0;
#                                         # then data/logs/score_s2a_done
#   tmux new -d -s score2a 'bash scripts/score_stage2.sh >> data/logs/score_stage2.log 2>&1'
# Env: PY, MODELS, CKPT, EVERY (default 400).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/tts/venv-voxcpm/bin/python}"
MODELS="${MODELS:-$HOME/tts/models}"
CKPT="${CKPT:-$HOME/tts/checkpoints/stage2_v2a}"
EVERY="${EVERY:-400}"
EV=data/eval
LOG=data/logs
B=$EV/refs/ref_male_b.wav
BASE=s0_voxcpm2_contB_freya40   # untrained model, same 40 Freya lines, same setting
P0REF=s2p_P0_plain_neutral      # Task 4's untrained plain run: voice reference for the style and tag runs
STYLES="angry happy sad"
TAGS="laugh breath"             # [ağlama] is out of run 2a
declare -A TR=([angry]="(öfkeyle)" [happy]="(neşeyle)" [sad]="(üzgün bir sesle)")

baseline() {  # the untrained run the plain numbers are compared with; made once
    [ -e "$LOG/${BASE}_done" ] && return
    "$PY" scripts/eval_tts.py all --system voxcpm2 --model-dir "$MODELS/VoxCPM2" --run "$BASE" \
        --sentences "$EV/freya_tr_eval.txt" --limit 40 --cfg 3.0 --prompt-wav "$B" --ref-wav "$B" --prompt "" \
        > "$LOG/$BASE.log" 2>&1
    "$PY" scripts/score_styles.py --run "$BASE" --target neutral >> "$LOG/$BASE.log" 2>&1
    "$PY" scripts/plain_metrics.py --run "$BASE" >> "$LOG/$BASE.log" 2>&1
    touch "$LOG/${BASE}_done"
}

one() {  # one <run> <sentences> <score_styles args> -- <eval_tts synth args...>
    local run=$1 sents=$2 sargs=()
    shift 2
    while [ "$1" != "--" ]; do sargs+=("$1"); shift; done
    shift
    "$PY" scripts/eval_tts.py all --system voxcpm2-lora --model-dir "$MODELS/VoxCPM2" --lora "$CK" \
        --lora-scale "$SCALE,0" --cfg 3.0 --run "$run" --sentences "$sents" "$@" > "$LOG/$run.log" 2>&1
    "$PY" scripts/score_styles.py --run "$run" "${sargs[@]}" >> "$LOG/$run.log" 2>&1
}

score() {  # score <step> <scale>
    local step=$1
    SCALE=$2
    CK=$(printf "%s/step_%07d" "$CKPT" "$step")
    [ -f "$CK/lora_weights.safetensors" ] || { echo "no checkpoint $CK"; return 1; }
    local tag="s2a_step${step}_x${SCALE}"
    if [ -e "$LOG/score_${tag}_done" ]; then echo "skip $tag (done)"; return; fi
    echo "== $tag  start $(date '+%F %T')"
    one "${tag}_plain" "$EV/freya_tr_eval.txt" --target neutral --voice-ref-run "$BASE" -- \
        --limit 40 --prompt-wav "$B" --ref-wav "$B" --prompt ""
    "$PY" scripts/plain_metrics.py --run "${tag}_plain" >> "$LOG/${tag}_plain.log" 2>&1
    for s in $STYLES; do
        for set in neutral matched; do
            f="$EV/style_$set.txt"; [ "$set" = matched ] && f="$EV/style_matched_$s.txt"
            one "${tag}_${s}_$set" "$f" --target "$s" --voice-ref-run "$P0REF" -- --ref-wav "$B" --prompt "${TR[$s]}"
        done
    done
    for t in $TAGS; do
        one "${tag}_${t}_tags" "$EV/tags_$t.txt" --target neutral --tag "$t" --voice-ref-run "$P0REF" -- \
            --ref-wav "$B" --prompt ""
    done
    touch "$LOG/score_${tag}_done"
    echo "== $tag  done  $(date '+%F %T')"
}

baseline
if [ $# -ge 1 ]; then
    score "$1" "${2:-1.0}"
    exit 0
fi
last=$(ls -d "$CKPT"/step_* | sed 's/.*step_0*//' | sort -n | tail -1)
for d in "$CKPT"/step_*; do
    n=$((10#${d##*step_}))
    [ "$n" -eq 0 ] && continue
    if [ $((n % EVERY)) -eq 0 ] || [ "$n" -eq "$last" ]; then score "$n" 1.0; fi
done
touch "$LOG/score_s2a_done"
echo "scoring done $(date '+%F %T')"
