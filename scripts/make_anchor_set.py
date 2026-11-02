#!/usr/bin/env python
"""Stage 2 anchor set (Task 5): plain sentences spoken by the base model in the stage 1 setting, mixed into the
LoRA's training data so plain speech stays as it is.

    # 1. sentences from the stage 2 tier (make_voxcpm_manifest.py --stage 2 writes <out>/tier_rows.jsonl)
    .venv/bin/python scripts/make_anchor_set.py sentences --tier data/voxcpm_stage2/tier_rows.jsonl --n 25000 \
        --out data/eval/anchor_sentences.txt
    # 2. synthesise + score them with eval_tts.py (run s2_anchor), then keep the good ones:
    .venv/bin/python scripts/make_anchor_set.py export --run s2_anchor --sentences data/eval/anchor_sentences.txt \
        --out data/anchor_base
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttsdata.anchors import anchor_keep, pick_sentences  # noqa: E402

EVAL_DIR = Path("data/eval")


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def cmd_sentences(a) -> None:
    rows = load_jsonl(Path(a.tier))
    out = Path(a.out)
    exclude = {l.strip() for p in sorted(EVAL_DIR.glob("*.txt")) if p.resolve() != out.resolve()
               for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}
    picked = pick_sentences(rows, exclude, a.n, a.seed)
    out.write_text("\n".join(picked) + "\n", encoding="utf-8")
    n_plain = sum(1 for r in rows if r.get("split") == "train" and not r.get("labelled"))
    print(f"{len(picked)} sentences of {a.n} wanted -> {out} ({n_plain} plain train rows in the tier, "
          f"{len(exclude)} eval lines excluded)")


def cmd_export(a) -> None:
    import librosa
    import numpy as np
    import soundfile as sf

    from ttsdata.build import _loudness_normalize
    from ttsdata.config import Config

    cfg = Config()
    run_dir = EVAL_DIR / a.run
    sentences = [l.strip() for l in Path(a.sentences).read_text(encoding="utf-8").splitlines() if l.strip()]
    scores = load_jsonl(run_dir / "scores.jsonl")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kept = 0
    for s in scores:
        if not anchor_keep(s, a.min_ref_cos):
            continue
        k = s["k"]
        y, sr = sf.read(run_dir / "wavs" / f"{k:04d}.wav", dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != a.sr:
            y, sr = librosa.resample(y, orig_sr=sr, target_sr=a.sr), a.sr
        y = _loudness_normalize(np.ascontiguousarray(y), sr, cfg.target_lufs, cfg.peak_ceiling)
        sf.write(out / f"{k:05d}.wav", y, sr, subtype="PCM_16")
        (out / f"{k:05d}.txt").write_text(sentences[k] + "\n", encoding="utf-8")
        kept += 1
    print(f"kept {kept} of {len(scores)} scored clips ({kept / max(1, len(scores)):.1%}) -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sentences")
    s.add_argument("--tier", default="data/voxcpm_stage2/tier_rows.jsonl")
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--out", default="data/eval/anchor_sentences.txt")
    s.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("export")
    e.add_argument("--run", default="s2_anchor")
    e.add_argument("--sentences", default="data/eval/anchor_sentences.txt")
    e.add_argument("--out", default="data/anchor_base")
    e.add_argument("--min-ref-cos", type=float, default=0.75)
    e.add_argument("--sr", type=int, default=24000)
    a = ap.parse_args()
    cmd_sentences(a) if a.cmd == "sentences" else cmd_export(a)


if __name__ == "__main__":
    main()
