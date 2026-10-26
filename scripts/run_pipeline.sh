#!/usr/bin/env bash
# Segment, verify and build everything in data/raw. Resumable: re-run after adding downloads.
set -euo pipefail
cd "$(dirname "$0")/.."
WORKERS="${WORKERS:-3}"
.venv/bin/ttsdata segment
scripts/verify_parallel.sh "$WORKERS"
.venv/bin/ttsdata build
.venv/bin/ttsdata speakers                              # speaker_id column from clustered ECAPA embeddings
.venv/bin/python scripts/audit_dataset.py --skip-asr   # integrity checks; drop --skip-asr for a Whisper spot-check
