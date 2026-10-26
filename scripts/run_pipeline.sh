#!/usr/bin/env bash
# Segment, verify and build everything in data/raw. Resumable: re-run after adding downloads.
set -euo pipefail
cd "$(dirname "$0")/.."
WORKERS="${WORKERS:-3}"
.venv/bin/ttsdata segment
scripts/verify_parallel.sh "$WORKERS"
.venv/bin/ttsdata build
.venv/bin/ttsdata speakers                              # speaker_id column from clustered ECAPA embeddings
.venv/bin/ttsdata merge-clips                           # longer clips from adjacent same-voice lines (data/dataset_long)
.venv/bin/ttsdata soundtags                             # subtitle sound descriptions -> style / non-verbal labels, flags
.venv/bin/ttsdata extend-events                         # clips re-cut to include the adjacent laugh / cry, tagged (data/dataset_events)
.venv/bin/ttsdata verify-events                         # Whisper + PANNs on the added audio: narrator out, sound confirmed
.venv/bin/python scripts/audit_dataset.py --skip-asr   # integrity checks; drop --skip-asr for a Whisper spot-check
