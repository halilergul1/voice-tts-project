"""Label speaker ids male / female from voice pitch (ttsdata.voicegender) -> <root>/speaker_gender.jsonl.

  speaker_gender.py --root data/dataset [--manifests data/voxcpm/train.jsonl,...] [--per-speaker 12] [--workers 8]

Measures up to --per-speaker random single clips per speaker id (wavs_clean when present, else wavs). With
--manifests only the speakers those manifests use (from data/dataset and data/dataset_long rows) are measured.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttsdata.separation import manifest_ids  # noqa: E402
from ttsdata.voicegender import clip_f0, speaker_gender  # noqa: E402


def _load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def _f0(path: str) -> float | None:
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return clip_f0(y, sr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/dataset")
    ap.add_argument("--long-root", default="data/dataset_long")
    ap.add_argument("--manifests", default="")
    ap.add_argument("--per-speaker", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root, long_root = Path(a.root), Path(a.long_root)
    rows = _load(root / "metadata.jsonl")
    speakers = None
    if a.manifests:
        mrows = [r for m in a.manifests.split(",") for r in _load(m)]
        ids = manifest_ids(mrows, root.name)
        long_ids = manifest_ids(mrows, long_root.name)
        speakers = {r.get("speaker_id") for r in rows if r["id"] in ids}
        if long_ids and (long_root / "metadata.jsonl").exists():
            speakers |= {r.get("speaker_id") for r in _load(long_root / "metadata.jsonl") if r["id"] in long_ids}
        speakers.discard(None)
    by_spk = defaultdict(list)
    for r in rows:
        s = r.get("speaker_id")
        if s and (speakers is None or s in speakers):
            by_spk[s].append(r["id"])
    rng = random.Random(a.seed)
    jobs = []
    for s, ids in sorted(by_spk.items()):
        rng.shuffle(ids)
        for i in ids[: a.per_speaker]:
            p = root / "wavs_clean" / f"{i}.wav"
            jobs.append((s, str(p if p.exists() else root / "wavs" / f"{i}.wav")))
    print(f"{len(by_spk)} speakers, {len(jobs)} clips to measure")
    f0s = defaultdict(list)
    with ProcessPoolExecutor(a.workers) as ex:
        for n, ((s, _), f) in enumerate(zip(jobs, ex.map(_f0, [p for _, p in jobs], chunksize=16)), 1):
            f0s[s].append(f)
            if n % 2000 == 0:
                print(f"[{n}/{len(jobs)}]", flush=True)
    out = root / "speaker_gender.jsonl"
    counts = Counter()
    with out.open("w", encoding="utf-8") as fh:
        for s in sorted(f0s):
            g = speaker_gender(f0s[s])
            counts[g["gender"]] += 1
            fh.write(json.dumps({"speaker_id": s, **g}, ensure_ascii=False) + "\n")
    print(f"wrote {out}: " + ", ".join(f"{k or 'unlabelled'} {v}" for k, v in counts.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
