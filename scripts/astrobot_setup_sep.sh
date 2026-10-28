#!/usr/bin/env bash
# Idempotent: the vocal-separation venv on astrobot (~/tts/venv-sep) for scripts/separate_vocals.py.
# Same torch build as the project venv (cu130, sees the RTX 5090), audio-separator on top; BS-Roformer runs on torch.
set -euo pipefail
UV="${UV:-$HOME/.local/bin/uv}"
VENV="${VENV:-$HOME/tts/venv-sep}"
TORCH="${TORCH:-2.14.1}"
mkdir -p "$HOME/tts"
[ -x "$VENV/bin/python" ] || "$UV" venv "$VENV" --python 3.12
"$UV" pip install --python "$VENV/bin/python" "torch==$TORCH" --index-url https://download.pytorch.org/whl/cu130
"$UV" pip install --python "$VENV/bin/python" "audio-separator[cpu]" librosa audioread soundfile pyloudnorm scipy "torch==$TORCH" \
    --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
"$VENV/bin/python" -c "import torch, audio_separator; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
