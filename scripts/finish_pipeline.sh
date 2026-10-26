#!/usr/bin/env bash
# Detached driver: verify everything still unverified (N workers), build, integrity audit, then drop a marker.
set -uo pipefail
cd "$(dirname "$0")/.."
WORKERS="${WORKERS:-3}"
scripts/verify_parallel.sh "$WORKERS" > data/logs/verify_batch.log 2>&1
echo "verify exit=$? $(date +%T)" >> data/logs/pipeline.log
.venv/bin/ttsdata build > data/logs/build.log 2>&1
echo "build exit=$? $(date +%T)" >> data/logs/pipeline.log
.venv/bin/python scripts/audit_dataset.py --skip-asr > data/logs/audit_quick.log 2>&1
echo "audit exit=$? $(date +%T)" >> data/logs/pipeline.log
touch data/logs/pipeline_done
