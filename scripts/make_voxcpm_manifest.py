"""Build VoxCPM fine-tuning manifests (train.jsonl / val.jsonl) from the audited dataset.

Strict tier (see ttsdata.manifest.tier_funnel), capped at --max-hours with a per-series share cap, 40 % of rows
paired with a same-speaker reference clip, optional replay rows (FLEURS-tr) with dataset_id 1, every row validated.
Merged clips from `ttsdata merge-clips` (--long-root) replace their member clips when they pass the tier; clips
flagged by `ttsdata soundtags` as overlapping / foreign / unintelligible / censored / TV-voice are dropped.
Also writes <out>/tier_report.md and data/eval/corpus_heldout.txt (val-set sentences for scripts/eval_tts.py).

usage: .venv/bin/python scripts/make_voxcpm_manifest.py --root data/dataset --out data/voxcpm --wavs wavs_clean \
          [--max-hours 60] [--ref-ratio 0.4] [--replay-dir data/fleurs_tr --replay-share 0.15] [--fetch-fleurs]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tarfile
from collections import defaultdict
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttsdata.manifest import (combine_units, manifest_row, pair_references, replay_text, tier_funnel,  # noqa: E402
                              validate_row)


def fetch_fleurs(out_dir: Path, min_dur: float = 3.0, max_dur: float = 20.0, target_lufs: float = -20.0) -> int:
    """Download the FLEURS Turkish train split (16 kHz wavs + tsv) from the HF hub, write 24 kHz clips + .txt."""
    from huggingface_hub import hf_hub_download
    from scipy.signal import resample_poly

    from ttsdata.separation import loudness_normalize

    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = Path(hf_hub_download("google/fleurs", "data/tr_tr/train.tsv", repo_type="dataset"))
    tar = Path(hf_hub_download("google/fleurs", "data/tr_tr/audio/train.tar.gz", repo_type="dataset"))
    text_of = {}
    for line in tsv.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            text_of[parts[1]] = parts[2]  # id, file name, raw transcription, ...
    n = 0
    with tarfile.open(tar) as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".wav"):
                continue
            name = Path(m.name).name
            if name not in text_of or ((out_dir / name).exists() and (out_dir / name).with_suffix(".txt").exists()):
                continue
            f = tf.extractfile(m)
            y, sr = sf.read(f, dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            dur = len(y) / sr
            if not (min_dur <= dur <= max_dur):
                continue
            if sr != 24000:
                g = gcd(sr, 24000)
                y = resample_poly(y, 24000 // g, sr // g).astype(np.float32)
            y = loudness_normalize(y, 24000, target_lufs, 0.99)
            (out_dir / name).with_suffix(".txt").write_text(text_of[name].strip() + "\n", encoding="utf-8")  # txt first
            sf.write(out_dir / name, y, 24000, subtype="PCM_16")
            n += 1
    return n


def cap_hours(rows: list[dict], max_hours: float, max_series_share: float, seed: int) -> list[dict]:
    """Seeded sample of rows totalling about max_hours, no series above max_series_share of it."""
    total = sum(r["duration"] for r in rows) / 3600
    if max_hours <= 0 or total <= max_hours:
        return rows
    rng = random.Random(seed)
    by_series: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_series[r["series"]].append(r)
    hours = {s: sum(r["duration"] for r in v) / 3600 for s, v in by_series.items()}
    alloc = {s: min(h, max_series_share * max_hours) for s, h in hours.items()}
    scale = max_hours / sum(alloc.values())
    if scale < 1:
        alloc = {s: a * scale for s, a in alloc.items()}
    out = []
    for s, v in by_series.items():
        v = sorted(v, key=lambda r: r["id"])
        rng.shuffle(v)
        acc = 0.0
        for r in v:
            if acc >= alloc[s] * 3600:
                break
            out.append(r)
            acc += r["duration"]
    return sorted(out, key=lambda r: r["id"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/dataset")
    ap.add_argument("--out", default="data/voxcpm")
    ap.add_argument("--wavs", default="wavs_clean")
    ap.add_argument("--long-root", default="data/dataset_long", help="merged clips (ttsdata merge-clips); '' to skip")
    ap.add_argument("--long-wavs", default="wavs_clean")
    ap.add_argument("--stage", type=int, default=1, choices=(1, 2), help="2 also drops clips under a laugh track")
    ap.add_argument("--max-hours", type=float, default=60.0)
    ap.add_argument("--max-series-share", type=float, default=0.4)
    ap.add_argument("--ref-ratio", type=float, default=0.4)
    ap.add_argument("--replay-dir", default="data/fleurs_tr")
    ap.add_argument("--replay-share", type=float, default=0.15)
    ap.add_argument("--fetch-fleurs", action="store_true")
    ap.add_argument("--heldout", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-cer", type=float, default=0.05)
    ap.add_argument("--min-cluster", type=int, default=10)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.fetch_fleurs:
        n = fetch_fleurs(Path(a.replay_dir))
        print(f"fleurs: {n} new clips in {a.replay_dir}")

    def load_jsonl(p):
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []

    def with_wavs(rs, wav_dir, what):
        present = {p.stem for p in wav_dir.glob("*.wav")} if wav_dir.exists() else set()
        kept = [r for r in rs if r["id"] in present]
        if len(kept) < len(rs):
            print(f"[warn] {len(rs) - len(kept)} {what} have no wav in {wav_dir}; dropped")
        return kept

    qfile = root / "quality.jsonl"
    if not qfile.exists():
        raise SystemExit(f"{qfile} missing: run scripts/audit_quality.py first")
    quality = {q["id"]: q for q in load_jsonl(qfile)}
    wav_dir = root / a.wavs
    singles = with_wavs(load_jsonl(root / "metadata.jsonl"), wav_dir, "clips")
    long_root = Path(a.long_root) if a.long_root else None
    merged = []
    if long_root and (long_root / "metadata.jsonl").exists():
        if not (long_root / "quality.jsonl").exists():
            raise SystemExit(f"{long_root / 'quality.jsonl'} missing: run scripts/audit_quality.py --root {long_root}")
        quality.update({q["id"]: q for q in load_jsonl(long_root / "quality.jsonl")})
        merged = with_wavs(load_jsonl(long_root / "metadata.jsonl"), long_root / a.long_wavs, "merged clips")
    else:
        print("[warn] no merged clips (run ttsdata merge-clips, or pass --long-root)")
    soundtags = None
    tag_files = [p / "soundtags.jsonl" for p in ([root] + ([long_root] if long_root else [])) if (p / "soundtags.jsonl").exists()]
    if tag_files:
        soundtags = {t["id"]: t for p in tag_files for t in load_jsonl(p)}
    else:
        print("[warn] no soundtags.jsonl: run ttsdata soundtags to drop overlapping / foreign / censored clips")
    tier_kw = dict(max_cer=a.max_cer, min_cluster=a.min_cluster, soundtags=soundtags, stage=a.stage)
    rows = combine_units(singles, merged, quality, **tier_kw)
    funnel = tier_funnel(rows, quality, **tier_kw)
    tier = funnel[-1][1]
    slice_rows = cap_hours(tier, a.max_hours, a.max_series_share, a.seed)
    train_rows = [r for r in slice_rows if r.get("split") != "val"]
    val_rows = [r for r in slice_rows if r.get("split") == "val"]
    pairs = {**pair_references(train_rows, a.ref_ratio, a.seed), **pair_references(val_rows, a.ref_ratio, a.seed)}
    by_id = {r["id"]: r for r in slice_rows}
    root_of = {"dataset": os.path.relpath(wav_dir, out),
               "dataset_long": os.path.relpath(long_root / a.long_wavs, out) if long_root else None}

    train, val = [], []
    for r in slice_rows:
        ref = by_id.get(pairs.get(r["id"]))
        d = manifest_row(r, ref, root_of[r["root"]], dataset_id=0, ref_audio_root=root_of[ref["root"]] if ref else None)
        (val if r.get("split") == "val" else train).append(d)

    replay_rows = []
    rdir = Path(a.replay_dir)
    if a.replay_share > 0 and rdir.exists():
        rng = random.Random(a.seed)
        files = sorted(rdir.glob("*.wav"))
        want = round(a.replay_share / (1 - a.replay_share) * len(train))
        rng.shuffle(files)
        skipped = 0
        for p in files:
            if len(replay_rows) >= want:
                break
            txt = p.with_suffix(".txt")
            text = replay_text(txt.read_text(encoding="utf-8")) if txt.exists() else None
            if text is None:  # digits, foreign words, markup: not readable as written
                skipped += 1
                continue
            replay_rows.append({"audio": f"{os.path.relpath(rdir, out)}/{p.name}", "text": text,
                                "duration": round(sf.info(p).duration, 3), "dataset_id": 1})
        print(f"replay: {len(replay_rows)} rows from {rdir} ({skipped} clips skipped: unreadable text)")
        train += replay_rows

    for d in train + val:
        errs = validate_row(d)
        if errs:
            raise SystemExit(f"invalid manifest row {d}: {errs}")
    rng = random.Random(a.seed)
    rng.shuffle(train)
    for name, data in (("train.jsonl", train), ("val.jsonl", val)):
        with (out / name).open("w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    eval_dir = Path("data/eval")
    eval_dir.mkdir(parents=True, exist_ok=True)
    held = sorted({r["normalized_text"] for r in slice_rows if r.get("split") == "val"})
    rng = random.Random(a.seed)
    rng.shuffle(held)
    (eval_dir / "corpus_heldout.txt").write_text("\n".join(held[: a.heldout]) + "\n", encoding="utf-8")

    def hrs(rs):
        return sum(r["duration"] for r in rs) / 3600

    n_merged_slice = sum(1 for r in slice_rows if r["root"] == "dataset_long")
    covered = sum(len(r["members"]) for r in rows if r["root"] == "dataset_long")
    lines = [f"# VoxCPM manifest tier report ({a.wavs}/, stage {a.stage})", "",
             f"single clips: {len(singles)} ({hrs(singles):.2f} h); merged clips: {len(merged)} ({hrs(merged):.2f} h), "
             f"of which {sum(1 for r in rows if r['root'] == 'dataset_long')} pass the tier and replace {covered} single clips",
             f"sound flags: {'applied (' + str(len(soundtags)) + ' tagged clips)' if soundtags is not None else 'not available'}",
             "", f"units entering the funnel: {len(rows)} ({hrs(rows):.2f} h)", "",
             "| gate | clips | hours |", "|---|---|---|"]
    lines += [f"| {name} | {len(rs)} | {hrs(rs):.2f} |" for name, rs in funnel]
    lines += ["", f"slice (max {a.max_hours} h, series cap {a.max_series_share:.0%}): {len(slice_rows)} clips, {hrs(slice_rows):.2f} h, "
              f"of which {n_merged_slice} merged ({hrs([r for r in slice_rows if r['root'] == 'dataset_long']):.2f} h)", "",
              "| series | clips | hours | speakers |", "|---|---|---|---|"]
    per = defaultdict(list)
    for r in slice_rows:
        per[r["series"]].append(r)
    for s, rs in sorted(per.items(), key=lambda kv: -hrs(kv[1])):
        lines.append(f"| {s} | {len(rs)} | {hrs(rs):.2f} | {len({r['speaker_id'] for r in rs})} |")
    n_ref = sum(1 for d in train + val if "ref_audio" in d)
    lines += ["", f"- rows with ref_audio: {n_ref} ({n_ref / max(1, len(slice_rows)):.0%} of corpus rows)",
              f"- replay rows (dataset_id 1): {len(replay_rows)} ({len(replay_rows) / max(1, len(train)):.0%} of train)",
              f"- train {len(train)} / val {len(val)}; held-out eval sentences: {min(a.heldout, len(held))}", ""]
    (out / "tier_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
