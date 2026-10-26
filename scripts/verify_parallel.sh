#!/usr/bin/env bash
# Run N `ttsdata verify` workers; each episode is claimed with a lock file, so workers share the queue.
# One Whisper decode loop does not saturate the Apple GPU: 2-3 workers give a real speed-up.
set -euo pipefail
cd "$(dirname "$0")/.."
N="${1:-3}"
mkdir -p data/logs
for ((k = 0; k < N; k++)); do
  .venv/bin/ttsdata verify >> "data/logs/verify_worker_$k.log" 2>&1 &
  sleep 20   # stagger so the workers pick different episodes
done
wait
echo "all verify workers finished"
