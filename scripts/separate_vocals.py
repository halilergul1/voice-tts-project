"""Strip background music from dataset clips where it is actually present.

Runs a vocal separator (audio-separator, BS-Roformer by default) over <root>/<wavs>/, measures the instrumental
stem against the vocal stem, and writes <root>/<out>/<id>.wav as either the original clip or the separated vocals
(choose_source), loudness-normalised like the build stage. Records music_db and source per clip in
<root>/quality.jsonl. Resumable: clips already present in <out>/ are skipped.

Run inside a venv with audio-separator (the trainer venv is not needed):
  .venv-sep/bin/python scripts/separate_vocals.py --root data/dataset [--limit 50] [--model <ckpt>] [--threshold-db -25]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttsdata.separation import choose_source, loudness_normalize, relative_loudness_db  # noqa: E402

DEFAULT_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


def _read(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def _write_quality(qfile: Path, quality: dict[str, dict]) -> None:
    tmp = qfile.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for q in quality.values():
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    os.replace(tmp, qfile)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if len(x) else 0.0


def _stems(output_files: list[str], cid: str) -> tuple[Path | None, Path | None]:
    voc = ins = None
    for f in output_files:
        name = Path(f).name
        if not name.startswith(cid + "_"):
            continue
        if "(Vocals)" in name:
            voc = Path(f)
        elif "(Instrumental)" in name:
            ins = Path(f)
    return voc, ins


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/dataset")
    ap.add_argument("--wavs", default="wavs")
    ap.add_argument("--out", default="wavs_clean")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--threshold-db", type=float, default=-10.0, help="instrumental re vocals; clean speech scores -12..-25")
    ap.add_argument("--target-lufs", type=float, default=-20.0)
    ap.add_argument("--peak", type=float, default=0.99)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.root)
    src_dir, out_dir = root / a.wavs, root / a.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in (root / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.limit:
        rows = rows[: a.limit]
    todo = [r["id"] for r in rows if not (out_dir / f"{r['id']}.wav").exists()]
    print(f"{len(rows)} clips, {len(todo)} to separate ({len(rows) - len(todo)} already in {out_dir})")

    qfile = root / "quality.jsonl"
    quality: dict[str, dict] = {}
    if qfile.exists():
        for l in qfile.read_text(encoding="utf-8").splitlines():
            if l.strip():
                q = json.loads(l)
                quality[q["id"]] = q

    if todo:
        from audio_separator.separator import Separator

        tmp = Path(tempfile.mkdtemp(prefix="sep_", dir=root))
        try:
            sep = Separator(output_dir=str(tmp), output_format="WAV", sample_rate=24000, log_level=40)
            sep.load_model(model_filename=a.model)
            music_dbs, n_sep = [], 0
            for b in range(0, len(todo), a.batch):
                ids = todo[b:b + a.batch]
                outputs = sep.separate([str(src_dir / f"{cid}.wav") for cid in ids])
                outputs = [str(tmp / f) if not Path(f).is_absolute() else f for f in outputs]
                for cid in ids:
                    y, sr = _read(src_dir / f"{cid}.wav")
                    voc_p, ins_p = _stems(outputs, cid)
                    vocal_ratio = None
                    if voc_p is None or ins_p is None:
                        print(f"[warn] {cid}: stems not found, keeping original")
                        music_db, source, chosen = None, "original", y
                    else:
                        voc, vsr = _read(voc_p)
                        ins, _ = _read(ins_p)
                        n = min(len(voc), len(ins))
                        music_db = round(relative_loudness_db(voc[:n], ins[:n]), 2)
                        vocal_ratio = round(_rms(voc) / max(_rms(y), 1e-9), 3)
                        source = choose_source(music_db, a.threshold_db, vocal_ratio)
                        chosen = voc if source == "separated" else y
                        if source == "separated" and vsr != sr:
                            from scipy.signal import resample_poly
                            from math import gcd

                            g = gcd(vsr, sr)
                            chosen = resample_poly(chosen, sr // g, vsr // g).astype(np.float32)
                    chosen = loudness_normalize(chosen, sr, a.target_lufs, a.peak)
                    sf.write(out_dir / f"{cid}.wav", chosen, sr, subtype="PCM_16")
                    q = quality.setdefault(cid, {"id": cid})
                    q["music_db"], q["source"], q["vocal_ratio"] = music_db, source, vocal_ratio
                    if music_db is not None:
                        music_dbs.append(music_db)
                    n_sep += source == "separated"
                for f in tmp.iterdir():
                    f.unlink()
                _write_quality(qfile, quality)  # after every batch: a killed run keeps music_db for what it wrote
                print(f"[{min(b + a.batch, len(todo))}/{len(todo)}] separated so far: {n_sep}")
            if music_dbs:
                md = np.array(music_dbs)
                print(f"music_db (instrumental - vocals): median {np.median(md):.1f} dB, p75 {np.percentile(md, 75):.1f}, "
                      f"p90 {np.percentile(md, 90):.1f}; {n_sep}/{len(todo)} clips use separated vocals "
                      f"(threshold {a.threshold_db} dB)")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    if quality:
        _write_quality(qfile, quality)
    return 0


if __name__ == "__main__":
    sys.exit(main())
