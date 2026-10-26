"""Audit a built dataset: file/metadata integrity, audio format, duration profile, and an independent
Whisper re-transcription of a random sample of FINAL clips (no word timestamps, no alignment) to
confirm that the clips really say what their text says.

usage: .venv/bin/python scripts/audit_dataset.py [--root data/dataset] [--n 50] [--seed 1] [--model REPO]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttsdata.asr import resolve_backend  # noqa: E402
from ttsdata.config import Config  # noqa: E402
from ttsdata.textnorm import charset_ok, for_cer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/dataset")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model", default=None, help="override whisper model (default: backend's large-v3-turbo)")
    ap.add_argument("--skip-asr", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    problems: list[str] = []

    rows = [json.loads(l) for l in (root / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    csv_lines = [l for l in (root / "metadata.csv").read_text(encoding="utf-8").splitlines() if l.strip()]
    wavs = sorted((root / "wavs").glob("*.wav"))
    ids = [r["id"] for r in rows]
    print(f"utterances: jsonl {len(rows)} | metadata.csv {len(csv_lines)} | wav files {len(wavs)}")
    if not (len(rows) == len(csv_lines) == len(wavs)):
        problems.append("metadata.jsonl / metadata.csv / wavs counts differ")
    if len(set(ids)) != len(ids):
        problems.append("duplicate ids")
    for l in csv_lines:
        parts = l.split("|")
        if len(parts) not in (3, 4) or not parts[1].strip() or not parts[2].strip():  # 4th column: speaker_id
            problems.append(f"bad metadata.csv line: {l[:80]}")
            break
    tr = len((root / "train.csv").read_text(encoding="utf-8").splitlines())
    va = len((root / "val.csv").read_text(encoding="utf-8").splitlines())
    if tr + va != len(rows):
        problems.append("train + val != total")
    bad_cs = [r["id"] for r in rows if not charset_ok(r["normalized_text"])]
    if bad_cs:
        problems.append(f"{len(bad_cs)} normalized_text with unexpected characters, e.g. {bad_cs[:3]}")

    durs, srs, chans, clipped, dc, missing = [], Counter(), Counter(), 0, 0.0, 0
    for r in rows:
        p = root / r["audio_file"]
        if not p.exists():
            missing += 1
            continue
        info = sf.info(p)
        srs[info.samplerate] += 1
        chans[info.channels] += 1
        y, sr = sf.read(p, dtype="float32")
        durs.append(len(y) / sr)
        clipped += int(np.mean(np.abs(y) > 0.985) > 0.001)
        dc = max(dc, abs(float(np.mean(y))))
        if abs(len(y) / sr - r["duration"]) > 0.02:
            problems.append(f"duration mismatch {r['id']}")
    if missing:
        problems.append(f"{missing} wav files missing")
    d = np.array(durs)
    print(f"sample rates {dict(srs)} | channels {dict(chans)} | clipped files {clipped} | max |DC| {dc:.4f}")
    print(f"duration s: total {d.sum() / 3600:.2f} h | mean {d.mean():.2f} | median {np.median(d):.2f} | "
          f"p5 {np.percentile(d, 5):.2f} | p95 {np.percentile(d, 95):.2f} | min {d.min():.2f} | max {d.max():.2f}")
    cps = np.array([len(r["normalized_text"]) / r["duration"] for r in rows])
    print(f"chars/sec: median {np.median(cps):.1f} | p5 {np.percentile(cps, 5):.1f} | p95 {np.percentile(cps, 95):.1f}")
    hist = Counter(min(int(x), 15) for x in d)
    print("duration histogram (s): " + " ".join(f"{k}:{hist[k]}" for k in sorted(hist)))
    print("per source: " + json.dumps(Counter(r["source"] for r in rows)))
    print("speaker_hint set: " + f"{sum(1 for r in rows if r['speaker_hint'])} / {len(rows)}")

    if not a.skip_asr and rows:
        import jiwer

        cfg = Config()
        if a.model:
            cfg.whisper_model = a.model
        backend, model = resolve_backend(cfg)
        print(f"\nASR backend: {backend} / {model}")
        rng = random.Random(a.seed)
        sample = rng.sample(rows, min(a.n, len(rows)))
        cers = []
        worst = []
        for r in sample:
            if backend == "mlx":
                import mlx_whisper
                hyp_text = mlx_whisper.transcribe(str(root / r["audio_file"]), path_or_hf_repo=model, language="tr",
                                                  temperature=0.0, condition_on_previous_text=False, verbose=None)["text"]
            else:
                from ttsdata.asr import _words_from_ct2
                hyp_text = _words_from_ct2(str(root / r["audio_file"]), model, cfg)[0]
            out = {"text": hyp_text}
            ref, hyp = for_cer(r["text"]), for_cer(out["text"])
            c = jiwer.cer(ref, hyp) if hyp else 1.0
            cers.append(c)
            worst.append((c, r["text"], out["text"].strip()))
        cers_np = np.array(cers)
        print(f"\nindependent re-transcription of {len(sample)} final clips: CER mean {cers_np.mean():.3f} | "
              f"median {np.median(cers_np):.3f} | exact {int((cers_np == 0).sum())} | >0.2: {int((cers_np > 0.2).sum())} "
              f"| >0.5: {int((cers_np > 0.5).sum())}")
        for c, t, h in sorted(worst, key=lambda x: -x[0])[:5]:
            print(f"  CER {c:.2f}  TEXT: {t}\n            ASR: {h}")
        if (cers_np > 0.5).mean() > 0.05:
            problems.append("more than 5% of sampled clips disagree strongly with their text")

    print("\nPROBLEMS:" if problems else "\nno integrity problems found")
    for p in problems:
        print(" -", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
