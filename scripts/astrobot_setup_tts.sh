#!/usr/bin/env bash
# Idempotent: the VoxCPM venv on astrobot (~/tts/venv-voxcpm) for scripts/eval_tts.py and the LoRA trainer, plus the
# VoxCPM2 and Trendyol-TTS snapshots under ~/tts/models and the upstream repo (trainer scripts, conf/) at ~/tts/VoxCPM.
# Same torch build as the project venv (cu130, sees the RTX 5090); voxcpm and the scoring stack on top. faster-whisper's
# CTranslate2 wants CUDA 12 cuBLAS/cuDNN: the cu12 wheels, preloaded by ttsdata.asr.
set -euo pipefail
UV="${UV:-$HOME/.local/bin/uv}"
VENV="${VENV:-$HOME/tts/venv-voxcpm}"
TORCH="${TORCH:-2.14.1}"
CU=https://download.pytorch.org/whl/cu130
mkdir -p "$HOME/tts/models" "$HOME/tts/checkpoints" "$HOME/tts/logs"
[ -x "$VENV/bin/python" ] || "$UV" venv "$VENV" --python 3.12
"$UV" pip install --python "$VENV/bin/python" "torch==$TORCH" torchaudio --index-url "$CU"
"$UV" pip install --python "$VENV/bin/python" voxcpm soundfile jiwer faster-whisper speechbrain num2words pyloudnorm \
    scipy tqdm pandas huggingface_hub "git+https://github.com/sarulab-speech/UTMOSv2.git" "torch==$TORCH" torchaudio \
    nvidia-cublas-cu12 nvidia-cudnn-cu12 \
    --extra-index-url "$CU" --index-strategy unsafe-best-match
"$VENV/bin/python" -c "import torch, voxcpm; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"$VENV/bin/python" - <<'EOF'
import os
from huggingface_hub import snapshot_download
for repo, name in [("openbmb/VoxCPM2", "VoxCPM2"), ("Trendyol/Trendyol-TTS", "Trendyol-TTS")]:
    print(repo, "->", snapshot_download(repo, local_dir=os.path.expanduser(f"~/tts/models/{name}")))
EOF
[ -d "$HOME/tts/VoxCPM/.git" ] || git clone https://github.com/OpenBMB/VoxCPM "$HOME/tts/VoxCPM"
ls "$HOME/tts/models/VoxCPM2" "$HOME/tts/models/Trendyol-TTS" "$HOME/tts/VoxCPM/conf" 2>&1 | head -30
