"""Quality audit of a built dataset, beyond integrity: per-clip silence edges, loudness, peaks, text problems,
optional UTMOSv2 naturalness, per-episode CER outliers, speaker coverage, and a listening folder.

Writes <root>/quality.jsonl (one row per clip; keeps music_db/source/vocal_ratio written by separate_vocals.py),
When --wavs names a folder that holds only some clips (wavs_clean after separate_vocals.py --manifests), the clips
found there are measured and reported; the others keep their earlier quality rows.
<root>/AUDIT.md and <root>/listen/ (symlinks + index.md). Exit code 1 when AUDIT.md lists problems.

usage: .venv/bin/python scripts/audit_quality.py [--root data/dataset] [--wavs wavs] [--utmos 0|1]
                                                 [--listen-per-series 5] [--listen-worst 20] [--workers 8] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttsdata.quality import episode_outliers, silence_edges, text_flags  # noqa: E402

BUCKETS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 15), (15, 1e9)]


def _clip_metrics(job: tuple[str, str, str, str]) -> dict:
    cid, path, text, normalized = job
    import pyloudnorm as pyln

    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    lead, trail = silence_edges(y, sr)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    try:
        lufs = float(pyln.Meter(sr).integrated_loudness(y.astype(np.float64)))
    except ValueError:
        lufs = float("nan")
    return {"id": cid, "lead_sil": round(lead, 3), "trail_sil": round(trail, 3), "peak": round(peak, 4),
            "clipped": bool(np.mean(np.abs(y) > 0.985) > 0.001) if len(y) else False,
            "lufs": None if np.isnan(lufs) else round(lufs, 2), "text_flags": text_flags(text, normalized)}


def _utmos_scores(wav_dir: Path, ids: list[str]) -> dict[str, float]:
    import torch
    import utmosv2

    device = "cuda" if torch.cuda.is_available() else "cpu"  # 'auto' assumes CUDA; MPS is not supported
    model = utmosv2.create_model(pretrained=True, device=device)
    res = model.predict(input_dir=str(wav_dir), device=device)  # predict() defaults to cuda:0 on its own
    out = {}
    for r in res:
        out[Path(r["file_path"]).stem] = float(r["predicted_mos"])
    return {i: out[i] for i in ids if i in out}


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/dataset")
    ap.add_argument("--wavs", default="wavs", help="wav folder under root to measure (wavs or wavs_clean)")
    ap.add_argument("--utmos", type=int, default=0)
    ap.add_argument("--listen-per-series", type=int, default=5)
    ap.add_argument("--listen-worst", type=int, default=20)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    root = Path(a.root)
    wav_dir = root / a.wavs
    if a.utmos:
        import utmosv2  # noqa: F401  fail fast: the UTMOS pass comes last and must not die after the metric pass

    integrity = subprocess.run([sys.executable, str(Path(__file__).with_name("audit_dataset.py")), "--root", str(root),
                                "--skip-asr"], capture_output=True, text=True)
    integrity_tail = [l for l in integrity.stdout.splitlines() if l.strip()][-3:]

    rows = [json.loads(l) for l in (root / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.limit:
        rows = rows[: a.limit]
    present = [r for r in rows if (wav_dir / f"{r['id']}.wav").exists()]
    if len(present) < len(rows):
        print(f"{len(rows) - len(present)} of {len(rows)} clips have no wav in {wav_dir}: not measured, earlier rows kept")
        rows = present
    stats = json.loads((root / "stats.json").read_text(encoding="utf-8")) if (root / "stats.json").exists() else {}
    previous: dict[str, dict] = {}
    qfile = root / "quality.jsonl"
    if qfile.exists():
        for l in qfile.read_text(encoding="utf-8").splitlines():
            if l.strip():
                q = json.loads(l)
                previous[q["id"]] = q

    jobs = [(r["id"], str(wav_dir / f"{r['id']}.wav"), r["text"], r["normalized_text"]) for r in rows]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        quality = list(ex.map(_clip_metrics, jobs, chunksize=64))
    by_id = {q["id"]: q for q in quality}
    for q in quality:
        prev = previous.get(q["id"], {})
        q["music_db"] = prev.get("music_db")
        q["source"] = prev.get("source")
        q["vocal_ratio"] = prev.get("vocal_ratio")
        q["utmos"] = prev.get("utmos")
    def write_quality():  # union with rows not recomputed in this run (e.g. --limit), atomically
        merged = {**previous, **by_id}
        tmp = qfile.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for q in merged.values():
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        os.replace(tmp, qfile)

    write_quality()
    if a.utmos:
        for cid, mos in _utmos_scores(wav_dir, [r["id"] for r in rows]).items():
            by_id[cid]["utmos"] = round(mos, 3)
        write_quality()

    # ---------------- report ----------------
    problems: list[str] = []
    n = len(rows)
    durs = np.array([r["duration"] for r in rows])
    hours = durs.sum() / 3600
    lines = [f"# Quality audit: {root}", "", f"{n} clips, {hours:.2f} h, measured on `{a.wavs}/`", "",
             "Integrity (`audit_dataset.py --skip-asr`):", "", "```", *integrity_tail, "```", ""]
    if integrity.returncode != 0:
        problems.append("audit_dataset.py reported integrity problems (see above)")

    lines += ["## Duration buckets", "", "| seconds | clips | hours | share of hours |", "|---|---|---|---|"]
    for lo, hi in BUCKETS:
        m = (durs >= lo) & (durs < hi)
        lines.append(f"| {lo}–{'∞' if hi > 1e8 else hi} | {int(m.sum())} | {durs[m].sum() / 3600:.2f} | {durs[m].sum() / durs.sum():.1%} |")
    short = (durs < 3).sum()
    lines += ["", f"Clips under 3 s (excluded from VoxCPM tier A): {short} ({short / n:.1%} of clips, "
                  f"{durs[durs < 3].sum() / durs.sum():.1%} of hours).", ""]

    flag_counts = Counter(fl for q in quality for fl in q["text_flags"])
    examples = defaultdict(list)
    text_of = {r["id"]: r["normalized_text"] for r in rows}
    for q in quality:
        for fl in q["text_flags"]:
            if len(examples[fl]) < 3:
                examples[fl].append(f"`{q['id']}`: {text_of[q['id']][:70]}")
    any_flag = sum(1 for q in quality if q["text_flags"])
    lines += ["## Text flags", "", f"{any_flag} clips ({any_flag / n:.1%}) carry at least one flag.", "",
              "| flag | clips | share | examples |", "|---|---|---|---|"]
    for fl, c in flag_counts.most_common():
        lines.append(f"| {fl} | {c} | {c / n:.1%} | {' · '.join(examples[fl])} |")
        if c / n > 0.05:
            problems.append(f"text flag `{fl}` on {c / n:.1%} of clips")

    lead = np.array([q["lead_sil"] for q in quality])
    trail = np.array([q["trail_sil"] for q in quality])
    tr_share = float((trail > 0.5).mean())
    lines += ["", "## Silence at the edges", "",
              f"- leading: median {np.median(lead):.2f} s, p95 {_pct(lead, 95):.2f} s, > 0.3 s on {(lead > 0.3).mean():.1%}",
              f"- trailing: median {np.median(trail):.2f} s, p95 {_pct(trail, 95):.2f} s, > 0.5 s on {tr_share:.1%} (VoxCPM wants < 0.5 s)", ""]
    if tr_share > 0.02:
        problems.append(f"trailing silence > 0.5 s on {tr_share:.1%} of clips")

    lufs = np.array([q["lufs"] for q in quality if q["lufs"] is not None])
    peak = np.array([q["peak"] for q in quality])
    clipped = sum(1 for q in quality if q["clipped"])
    lines += ["## Loudness and peaks", "",
              f"- integrated loudness: mean {lufs.mean():.1f} LUFS, p5 {_pct(lufs, 5):.1f}, p95 {_pct(lufs, 95):.1f} ({len(lufs)} measurable)",
              f"- peak: median {np.median(peak):.2f}, p99 {_pct(peak, 99):.2f}; clipped clips: {clipped} ({clipped / n:.2%})", ""]
    if clipped / n > 0.005:
        problems.append(f"{clipped / n:.2%} of clips clip")

    outl = episode_outliers(rows)
    lines += ["## Episode CER outliers", "", "| uid | mean CER | clips |", "|---|---|---|"]
    lines += [f"| {o['uid']} | {o['mean_cer']} | {o['n']} |" for o in outl] or ["| none | | |"]
    for o in outl:
        if o["mean_cer"] > 0.08:
            problems.append(f"episode {o['uid']} mean CER {o['mean_cer']}")

    sp = stats.get("speakers", {}).get("per_series", {})
    lines += ["", "## Speaker clusters per series", "", "| series | clips | ids | ids ≥ 10 clips | clips covered |", "|---|---|---|---|---|"]
    for s, v in sp.items():
        lines.append(f"| {s} | {v.get('clips')} | {v.get('speakers')} | {v.get('speakers_ge10')} | {v.get('coverage_ge10')} |")
    if not sp:
        lines.append("| (run `ttsdata speakers` first) | | | | |")

    series_of = {r["id"]: r["series"] for r in rows}
    ut = [(series_of[q["id"]], q["utmos"]) for q in quality if q.get("utmos") is not None]
    if ut:
        lines += ["", "## UTMOSv2 (naturalness proxy, English-trained)", "", "| series | clips | mean | p10 |", "|---|---|---|---|"]
        per = defaultdict(list)
        for s, m in ut:
            per[s].append(m)
        for s, v in sorted(per.items()):
            lines.append(f"| {s} | {len(v)} | {np.mean(v):.2f} | {_pct(v, 10):.2f} |")
    md = np.array([q["music_db"] for q in quality if q.get("music_db") is not None])
    if len(md):
        sep = sum(1 for q in quality if q.get("source") == "separated")
        lines += ["", "## Music under speech (from separate_vocals.py)", "",
                  f"- instrumental minus vocal loudness: median {np.median(md):.1f} dB, p90 {_pct(md, 90):.1f} dB; "
                  f"{sep} clips ({sep / len(md):.1%}) use the separated vocals", ""]

    # ---------------- listening folder ----------------
    listen = root / "listen"
    listen.mkdir(exist_ok=True)
    for p in listen.iterdir():
        if p.is_symlink() or p.suffix == ".wav":
            p.unlink()
    rng = random.Random(a.seed)
    picks: list[tuple[str, str, str]] = []  # tag, id, why
    per_series = defaultdict(list)
    for r in rows:
        per_series[r["series"]].append(r["id"])
    for s, ids in sorted(per_series.items()):
        for cid in rng.sample(ids, min(a.listen_per_series, len(ids))):
            picks.append(("series", cid, s))
    if ut:
        worst = sorted((q for q in quality if q.get("utmos") is not None), key=lambda q: q["utmos"])[: a.listen_worst]
        picks += [("worst_utmos", q["id"], f"utmos {q['utmos']}") for q in worst]
    for q in sorted(quality, key=lambda q: -q["trail_sil"])[:10]:
        picks.append(("trail", q["id"], f"trailing silence {q['trail_sil']} s"))
    for q in [q for q in quality if q["text_flags"]][:10]:
        picks.append(("flag", q["id"], ",".join(q["text_flags"])))
    idx = ["# Listening sample", "", "| file | why | text |", "|---|---|---|"]
    for k, (tag, cid, why) in enumerate(picks):
        name = f"{k:03d}_{tag}_{cid}.wav"
        os.symlink(os.path.relpath(wav_dir / f"{cid}.wav", listen), listen / name)
        idx.append(f"| `{name}` | {why} | {text_of[cid]} |")
    (listen / "index.md").write_text("\n".join(idx) + "\n", encoding="utf-8")

    lines += ["", "## PROBLEMS", ""] + ([f"- {p}" for p in problems] or ["none"]) + ["",
              f"Listening sample: `{listen}/index.md` ({len(picks)} clips).", ""]
    (root / "AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
