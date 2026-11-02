#!/usr/bin/env python
"""Flag event clips whose sound is not the line speaker's own or is covered by the audio-description narrator
(stage 2 Task 3C). CPU only. Writes <events-root>/event_checks.jsonl; make_voxcpm_manifest.py --event-checks reads it.

    .venv/bin/python scripts/check_events.py
    .venv/bin/python scripts/check_events.py --limit 50 --out /tmp/ev.jsonl     # quick look
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttsdata.eventclean import CHECKS, narrator_overlap, pitch_conflict, who_conflict  # noqa: E402
from ttsdata.narration import name_clusters  # noqa: E402
from ttsdata.voicegender import clip_f0  # noqa: E402


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def ext_f0(job: tuple[str, float, float]) -> float | None:
    """Median F0 of the extension part (seconds from clip start) of one clip."""
    path, a, b = job
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        return None
    if y.ndim > 1:
        y = y.mean(axis=1)
    seg = y[max(0, int(a * sr)): int(b * sr)]
    return clip_f0(np.ascontiguousarray(seg), sr) if len(seg) >= int(0.2 * sr) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-root", default="data/dataset_events")
    ap.add_argument("--wavs", default="wavs_clean")
    ap.add_argument("--root", default="data/dataset", help="line clips: metadata.jsonl for name clusters, "
                    "speaker_gender.jsonl, narration.jsonl")
    ap.add_argument("--min-narr-sec", type=float, default=0.3)
    ap.add_argument("--male-max", type=float, default=260.0)
    ap.add_argument("--female-min", type=float, default=140.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    ev_root, root = Path(a.events_root), Path(a.root)
    events = [r for r in load_jsonl(ev_root / "metadata.jsonl") if r.get("event")][: a.limit or None]
    spans = defaultdict(list)
    for s in load_jsonl(root / "narration.jsonl"):
        spans[s["uid"]].append((s["start"], s["end"]))
    clusters = name_clusters(load_jsonl(root / "metadata.jsonl"))
    gender = {g["speaker_id"]: g.get("gender") for g in load_jsonl(root / "speaker_gender.jsonl")}
    print(f"{len(events)} event clips, narrator spans for {len(spans)} episodes, {len(clusters)} named clusters, "
          f"{sum(1 for v in gender.values() if v)} speakers with a gender", flush=True)

    jobs, need = [], []
    for r in events:
        if gender.get(r.get("speaker_id")):
            e, t0 = r["event"], r["orig_start"]
            path = ev_root / a.wavs / f"{r['id']}.wav"
            if not path.exists():
                path = ev_root / r["audio_file"]
            jobs.append((str(path), e["start"] - t0, e["end"] - t0))
            need.append(r["id"])
    with Pool(a.workers) as pool:
        f0 = dict(zip(need, pool.map(ext_f0, jobs, chunksize=16)))

    out = Path(a.out) if a.out else ev_root / "event_checks.jsonl"
    flagged, per_tag = Counter(), defaultdict(Counter)
    with out.open("w", encoding="utf-8") as f:
        for r in events:
            e, g = r["event"], gender.get(r.get("speaker_id"))
            row = {"id": r["id"], "tag": r["tag"], "verified": r.get("verified"),
                   "narrator": narrator_overlap(e, spans.get(r["uid"], []), a.min_narr_sec),
                   "who_conflict": who_conflict(e.get("who"), r["series"], r.get("speaker_id"), clusters),
                   "pitch_conflict": pitch_conflict(f0.get(r["id"]), g, a.male_max, a.female_min),
                   "ext_f0": None if f0.get(r["id"]) is None else round(f0[r["id"]], 1), "line_gender": g}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            per_tag[r["tag"]]["clips"] += 1
            for k in CHECKS:
                per_tag[r["tag"]][k] += bool(row[k])
            if any(row[k] for k in CHECKS):
                flagged[r["tag"]] += 1
                per_tag[r["tag"]]["flagged"] += 1
    print(f"wrote {out}")
    print("| tag | clips | narrator | who_conflict | pitch_conflict | flagged | kept |")
    print("|---|---|---|---|---|---|---|")
    for tag, c in sorted(per_tag.items(), key=lambda kv: -kv[1]["clips"]):
        print(f"| {tag} | {c['clips']} | {c['narrator']} | {c['who_conflict']} | {c['pitch_conflict']} | "
              f"{c['flagged']} | {c['clips'] - c['flagged']} |")


if __name__ == "__main__":
    main()
