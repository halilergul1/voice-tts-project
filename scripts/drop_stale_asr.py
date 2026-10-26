"""Remove asr.jsonl records produced by an older evaluate() (no `edge_unmatched` key) so `verify` redoes them.
Skips episodes currently locked by a running worker."""
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/segments")
for asr in root.glob("*/*/asr.jsonl"):
    lock = asr.with_name(".verify.lock")
    if lock.exists():
        try:
            os.kill(int(lock.read_text().strip()), 0)
            print(f"[locked] {asr.parent.name}")
            continue
        except (OSError, ValueError):
            pass
    keep, dropped = [], 0
    for line in asr.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1
            continue
        if "edge_unmatched" in r or r.get("error") or r.get("trim_start") is None:
            keep.append(line)
        else:
            dropped += 1
    if dropped:
        asr.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
    print(f"{asr.parent.name}: kept {len(keep)} dropped {dropped}")
