#!/usr/bin/env bash
# Stage 2 Task 4: what untrained VoxCPM2 already does with speaking styles and sound tags, all in clip B's voice, cfg 3.0.
#   P0    plain S1: continue from clip B and clone it, no prefix (the stage 1 setting; baseline)
#   P1    S1 + Turkish prefix, e.g. "(öfkeyle)"
#   P2    clone clip B only + Turkish prefix
#   P3    clone clip B only + English prefix, e.g. "(An angry voice)"
#   P4    continue from a clip spoken in the style (data/eval/refs/style_<label>.wav, chosen by ear), clone clip B
#   tags  Turkish tag inline ("[gülme] ...") in P1 and P2; English tag inline ("[laughter] ...") in P2en
# Styles run on data/eval/style_neutral.txt (20 emotion-free sentences) and style_matched_<label>.txt (10 that fit the
# style); tags on tags_<tag>.txt / tags_<tag>_en.txt (10 each, tag at the start in 5, at the end in 5); P0 also on
# tags_plain.txt (the same 30 sentences without tags). Runs are named s2p_<setup>_<label>_<set>: eval_tts.py all, then
# score_styles.py (voice similarity against s2p_P0_plain_neutral).
#   tmux new -d -s s2probe 'scripts/probe_stage2.sh >> data/logs/s2_probe.log 2>&1'
# Env: SETUPS (default "P0 P1 P2 P3 tags P4"), PY, MODELS. A P4 run waits (is skipped) until its style clip exists.
# Resumable: data/logs/s2p_<run>_done per run; every requested run done: data/logs/s2_probe_done.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/tts/venv-voxcpm/bin/python}"
MODELS="${MODELS:-$HOME/tts/models}"
SETUPS="${SETUPS:-P0 P1 P2 P3 tags P4}"
B=data/eval/refs/ref_male_b.wav
EV=data/eval
LOG=data/logs
P0REF=s2p_P0_plain_neutral
STYLES="angry happy sad"
TAGS="laugh breath cry"
declare -A TR=([angry]="(öfkeyle)" [happy]="(neşeyle)" [sad]="(üzgün bir sesle)")
declare -A EN=([angry]="(An angry voice)" [happy]="(A happy, cheerful voice)" [sad]="(A sad voice)")
missing=0

probe() {  # probe <run> <target style> <tag or -> <sentences> <eval_tts.py synth args...>
    local run=$1 target=$2 tag=$3 sents=$4 sargs=()
    shift 4
    if [ -e "$LOG/${run}_done" ]; then echo "skip $run (done)"; return; fi
    echo "== $run  start $(date '+%F %T')"
    "$PY" scripts/eval_tts.py all --system voxcpm2 --model-dir "$MODELS/VoxCPM2" --run "$run" --sentences "$sents" \
        --cfg 3.0 "$@" > "$LOG/$run.log" 2>&1
    [ "$tag" != - ] && sargs+=(--tag "$tag")
    [ "$run" != "$P0REF" ] && sargs+=(--voice-ref-run "$P0REF")
    "$PY" scripts/score_styles.py --run "$run" --target "$target" "${sargs[@]}" >> "$LOG/$run.log" 2>&1
    touch "$LOG/${run}_done"
    echo "== $run  done  $(date '+%F %T')"
}

styles() {  # styles <setup> <style> <synth args...>: the neutral and the matched sentence set
    local setup=$1 s=$2
    shift 2
    probe "s2p_${setup}_${s}_neutral" "$s" - "$EV/style_neutral.txt" "$@"
    probe "s2p_${setup}_${s}_matched" "$s" - "$EV/style_matched_$s.txt" "$@"
}

for setup in $SETUPS; do
    case $setup in
    P0)
        probe "$P0REF" neutral - "$EV/style_neutral.txt" --prompt-wav "$B" --ref-wav "$B" --prompt ""
        for s in $STYLES; do
            probe "s2p_P0_${s}_matched" "$s" - "$EV/style_matched_$s.txt" --prompt-wav "$B" --ref-wav "$B" --prompt ""
        done
        probe s2p_P0_plain_tags neutral - "$EV/tags_plain.txt" --prompt-wav "$B" --ref-wav "$B" --prompt "" ;;
    P1) for s in $STYLES; do styles P1 "$s" --prompt-wav "$B" --ref-wav "$B" --prompt "${TR[$s]}"; done ;;
    P2) for s in $STYLES; do styles P2 "$s" --ref-wav "$B" --prompt "${TR[$s]}"; done ;;
    P3) for s in $STYLES; do styles P3 "$s" --ref-wav "$B" --prompt "${EN[$s]}"; done ;;
    P4)
        for s in $STYLES; do
            clip="$EV/refs/style_$s.wav"
            if [ ! -f "$clip" ]; then echo "skip P4 $s: no $clip yet (pick one by ear first)"; missing=1; continue; fi
            styles P4 "$s" --prompt-wav "$clip" --ref-wav "$B" --prompt ""
        done ;;
    tags)
        for t in $TAGS; do
            probe "s2p_P1_${t}_tags" neutral "$t" "$EV/tags_$t.txt" --prompt-wav "$B" --ref-wav "$B" --prompt ""
            probe "s2p_P2_${t}_tags" neutral "$t" "$EV/tags_$t.txt" --ref-wav "$B" --prompt ""
            probe "s2p_P2en_${t}_tags" neutral "$t" "$EV/tags_${t}_en.txt" --ref-wav "$B" --prompt ""
        done ;;
    *) echo "unknown setup $setup"; exit 2 ;;
    esac
done
if [ "$missing" -eq 0 ]; then touch "$LOG/s2_probe_done"; fi
echo "probe pass done $(date '+%F %T')$([ "$missing" -eq 1 ] && echo ', P4 still waiting for style clips')"
