"""Strip background music from dataset clips where it is actually present.

Runs a vocal separator (audio-separator, BS-Roformer by default) over <root>/<wavs>/, measures the instrumental
stem against the vocal stem, and writes <root>/<out>/<id>.wav as either the original clip or the separated vocals
(choose_source), loudness-normalised like the build stage. Records music_db and source per clip in
<root>/quality.jsonl. Resumable: clips already present in <out>/ are skipped. --manifests limits the run to the clips those VoxCPM
manifests use from this root (audio + ref_audio), e.g. the 60 h training slices instead of the whole build.

Run inside a venv with audio-separator (the trainer venv is not needed):
  .venv-sep/bin/python scripts/separate_vocals.py --root data/dataset [--limit 50] [--model <ckpt>] [--threshold-db -25]
      [--manifests data/voxcpm/train.jsonl,data/voxcpm/val.jsonl] [--shard K/N]
Parallel: N processes with --shard 0/N .. N-1/N each clean every N-th clip and append their results to
<root>/separation.shard<K>of<N>.jsonl instead of rewriting quality.jsonl; then one --merge run folds those files into
quality.jsonl and removes them. (One process keeps the GPU mostly idle: short clips are bound by file handling.)
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
from ttsdata.separation import (alias, choose_source, loudness_normalize, manifest_ids, merge_separation,  # noqa: E402
                                relative_loudness_db, shard, stems_for)

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
    ap.add_argument("--manifests", default="", help="comma-separated manifest jsonl files: only the clips they use")
    ap.add_argument("--shard", default="", help="K/N: clean part K of N (0-based) and write a shard result file")
    ap.add_argument("--merge", action="store_true", help="fold <root>/separation.shard*.jsonl into quality.jsonl and exit")
    a = ap.parse_args()
    root = Path(a.root)
    src_dir, out_dir = root / a.wavs, root / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    qfile = root / "quality.jsonl"
    if a.merge:
        files = sorted(root.glob("separation.shard*.jsonl"))
        rows = [json.loads(l) for f in files for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        quality = {}
        if qfile.exists():
            quality = {q["id"]: q for q in map(json.loads, filter(str.strip, qfile.read_text(encoding="utf-8").splitlines()))}
        _write_quality(qfile, merge_separation(quality, rows))
        for f in files:
            f.unlink()
        n_sep = sum(1 for r in rows if r.get("source") == "separated")
        print(f"merged {len(rows)} clips from {len(files)} shard files into {qfile} ({n_sep} use separated vocals)")
        return 0

    rows = [json.loads(l) for l in (root / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.manifests:
        used = set()
        for m in a.manifests.split(","):
            used |= manifest_ids([json.loads(l) for l in Path(m).read_text(encoding="utf-8").splitlines() if l.strip()],
                                 root.name)
        rows = [r for r in rows if r["id"] in used]
        print(f"--manifests: {len(rows)} clips of {root} are used by {a.manifests}")
    if a.limit:
        rows = rows[: a.limit]
    todo = [r["id"] for r in rows if not (out_dir / f"{r['id']}.wav").exists()]
    shard_file = None
    if a.shard:
        k, n = (int(x) for x in a.shard.split("/"))
        todo = shard(todo, k, n)
        shard_file = (root / f"separation.shard{k}of{n}.jsonl").open("a", encoding="utf-8")
    print(f"{len(rows)} clips, {len(todo)} to separate ({len(rows) - len(todo)} already in {out_dir})")

    quality: dict[str, dict] = {}
    if qfile.exists() and not shard_file:  # a shard worker only appends its own results
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
                links = tmp / "in"
                links.mkdir(exist_ok=True)
                for j, cid in enumerate(ids):
                    os.symlink((src_dir / f"{cid}.wav").resolve(), links / f"{alias(j)}.wav")
                outputs = sep.separate([str(links / f"{alias(j)}.wav") for j in range(len(ids))])
                outputs = [str(tmp / f) if not Path(f).is_absolute() else f for f in outputs]
                for j, cid in enumerate(ids):
                    y, sr = _read(src_dir / f"{cid}.wav")
                    voc_p, ins_p = stems_for(outputs, alias(j))
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
                    if shard_file:
                        shard_file.write(json.dumps({"id": cid, "music_db": music_db, "source": source,
                                                     "vocal_ratio": vocal_ratio}) + "\n")
                        shard_file.flush()
                    if music_db is not None:
                        music_dbs.append(music_db)
                    n_sep += source == "separated"
                for f in links.iterdir():
                    f.unlink()
                for f in tmp.iterdir():
                    if f.is_file():
                        f.unlink()
                if not shard_file:
                    _write_quality(qfile, quality)  # after every batch: a killed run keeps music_db for what it wrote
                print(f"[{min(b + a.batch, len(todo))}/{len(todo)}] separated so far: {n_sep}")
            if music_dbs:
                md = np.array(music_dbs)
                print(f"music_db (instrumental - vocals): median {np.median(md):.1f} dB, p75 {np.percentile(md, 75):.1f}, "
                      f"p90 {np.percentile(md, 90):.1f}; {n_sep}/{len(todo)} clips use separated vocals "
                      f"(threshold {a.threshold_db} dB)")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    if shard_file:
        shard_file.close()  # quality.jsonl is left to the --merge run
    elif quality:
        _write_quality(qfile, quality)
    return 0


if __name__ == "__main__":
    sys.exit(main())
