"""Cut training clips from the episode audio again, with --ctx seconds of context on each side, for Sidon restoration
(ttsdata.restoration). Clips: those the vocal separation cleaned (rows of <root>/quality.jsonl with a source); with
--manifests, only those of them the manifests use (as separate_vocals.py --manifests picks them), so clips another
selection separated earlier do not go on to Sidon.

  .venv/bin/python scripts/cut_context.py --root data/dataset [--ctx 0.5] [--workers 8] [--out wavs_ctx]
      [--manifests data/voxcpm_stage2_rawlabels/train.jsonl,data/voxcpm_stage2_rawlabels/val.jsonl]

Writes <root>/<out>/<id>.wav (24 kHz mono) and <root>/ctx.jsonl rows {id, lead, dur}: lead = seconds of context
before the clip, dur = the clip's own length. Each episode is decoded once. Resumable: ids already in ctx.jsonl with a
wav are skipped, and so are clips Sidon already restored (a wav in <root>/<--restored>), so the context folders can be
deleted after restoration.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttsdata.restoration import context_span, episode_audio, ids_to_cut  # noqa: E402
from ttsdata.separation import manifest_clip_ids  # noqa: E402

SR = 24_000


def _load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def _decode(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def _cut_episode(job: tuple[str, list[dict], str, float]) -> list[dict]:
    audio, clips, out, ctx = job
    y = _decode(Path(audio))
    total = len(y) / SR
    rows = []
    for c in clips:
        t0, t1, lead = context_span(c["orig_start"], c["orig_end"], ctx, total)
        seg = y[int(round(t0 * SR)): int(round(t1 * SR))]
        if len(seg) == 0:
            continue
        sf.write(Path(out) / f"{c['id']}.wav", np.clip(seg, -1, 1), SR, subtype="PCM_16")
        rows.append({"id": c["id"], "lead": lead, "dur": round(c["orig_end"] - c["orig_start"], 3)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/dataset")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="wavs_ctx")
    ap.add_argument("--ctx", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N clips only (testing)")
    ap.add_argument("--manifests", default="", help="comma-separated manifests: only the separated clips they use")
    ap.add_argument("--restored", default="wavs_sidon", help="Sidon output folder: clips with a wav there are skipped")
    a = ap.parse_args()
    root, raw = Path(a.root), Path(a.raw)
    out = root / a.out
    out.mkdir(parents=True, exist_ok=True)
    used = {q["id"] for q in _load(root / "quality.jsonl") if q.get("source")}
    if a.manifests:
        n_sep = len(used)
        used &= manifest_clip_ids(root, a.manifests.split(","))
        print(f"--manifests: {len(used)} of {n_sep} separated clips are used by {a.manifests}", flush=True)
    meta = [r for r in _load(root / "metadata.jsonl") if r["id"] in used]
    if a.limit:
        meta = meta[: a.limit]
    ctx_file = root / "ctx.jsonl"
    cut = {r["id"] for r in _load(ctx_file) if (out / f"{r['id']}.wav").exists()}
    restored = {p.stem for p in (root / a.restored).glob("*.wav")}
    todo = set(ids_to_cut([r["id"] for r in meta], cut, restored))
    done = [r["id"] for r in meta if r["id"] not in todo]
    by_ep: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in meta:
        if r["id"] in todo:
            by_ep[(r["source"], r["uid"])].append(r)
    jobs, missing = [], 0
    for (source, uid), clips in by_ep.items():
        audio = episode_audio(raw / source / uid)
        if audio is None:
            missing += len(clips)
            continue
        jobs.append((str(audio), clips, str(out), a.ctx))
    print(f"{len(meta)} clips, {len(done)} done, {sum(len(j[1]) for j in jobs)} to cut from {len(jobs)} episodes"
          f"{f', {missing} without episode audio' if missing else ''}", flush=True)
    n = 0
    with ProcessPoolExecutor(a.workers) as ex, ctx_file.open("a", encoding="utf-8") as fh:
        for i, fut in enumerate(as_completed([ex.submit(_cut_episode, j) for j in jobs]), 1):
            for row in fut.result():
                fh.write(json.dumps(row) + "\n")
                n += 1
            fh.flush()
            if i % 50 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)} episodes] {n} clips cut", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
