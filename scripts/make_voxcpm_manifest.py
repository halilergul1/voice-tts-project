"""Build VoxCPM fine-tuning manifests (train.jsonl / val.jsonl) from the audited dataset.

Strict tier (see ttsdata.manifest.tier_funnel), capped at --max-hours with a per-series share cap, 40 % of rows
paired with a same-speaker reference clip, optional replay rows (FLEURS-tr) with dataset_id 1, every row validated.
Merged clips from `ttsdata merge-clips` (--long-root) replace their member clips when they pass the tier; clips
flagged by `ttsdata soundtags` as overlapping / foreign / unintelligible / censored / TV-voice are dropped.
--stage 1 (prosody): plain text, no event-extended rows, no rows whose subtitle label marks a strong delivery.
--stage 2 (expressive): verified event-extended rows (--events-root, --event-evidence) replace their base clips,
Turkish style prefixes "(ağlayarak)" and inline tags "[gülme]" are written for labels frequent enough in the slice
(--min-style / --min-tag), labelled rows are oversampled (--label-repeat, capped by --label-max-share).
Stage 2 slice: labelled clips first (--label-first, rarest label first, --label-quota per label), then a random
fill of unlabelled clips, so a 60 h sample of a ~350 h tier keeps the rare event tags.
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
from collections import Counter, defaultdict
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttsdata.labels import EVENT_TAG, STYLE_PREFIX, allowed_labels, auto_styles, labelled_text, oversample  # noqa: E402
from ttsdata.manifest import (cap_hours, combine_events, combine_units, labels_first, manifest_row,  # noqa: E402
                              pair_references, replay_text, restore_bases,
                              stage_filter, tier_funnel, validate_row)


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/dataset")
    ap.add_argument("--out", default="data/voxcpm")
    ap.add_argument("--wavs", default="wavs_clean")
    ap.add_argument("--long-root", default="data/dataset_long", help="merged clips (ttsdata merge-clips); '' to skip")
    ap.add_argument("--long-wavs", default="wavs_clean")
    ap.add_argument("--stage", type=int, default=1, choices=(1, 2), help="1 plain prosody; 2 labels + event clips")
    ap.add_argument("--events-root", default="data/dataset_events", help="event-extended clips (ttsdata extend-events); '' to skip")
    ap.add_argument("--events-wavs", default="wavs_clean")
    ap.add_argument("--event-evidence", default="laugh_syllables,tagger,clean", help="accepted verification evidence levels")
    ap.add_argument("--include-unconfirmed", action="store_true", help="also take unconfirmed event clips (stage 2)")
    ap.add_argument("--min-style", type=int, default=50, help="clips a style needs in the slice to be written as a prefix")
    ap.add_argument("--min-tag", type=int, default=300, help="instances an event tag needs in the slice to be written")
    ap.add_argument("--label-repeat", type=int, default=3, help="stage 2: labelled rows repeated this many times")
    ap.add_argument("--label-max-share", type=float, default=0.4, help="stage 2: labelled rows at most this share of train")
    ap.add_argument("--label-first", action=argparse.BooleanOptionalAction, default=True,
                    help="stage 2: take labelled clips first (rarest label first), then fill with unlabelled clips at random")
    ap.add_argument("--label-quota", type=int, default=3000, help="stage 2 --label-first: clips taken first per label")
    ap.add_argument("--auto-styles", action="store_true", help="stage 2: angry / sad / happy prefixes from ttsdata emotions (emotions.jsonl)")
    ap.add_argument("--auto-min-conf", type=float, default=0.6, help="confidence a clip needs for an automatic style")
    ap.add_argument("--auto-conf", default="", help="per-class overrides from the calibration report; 2026-10-27: angry=0.70,sad=0.85,happy=0.99")
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
    units = rows  # kept so an event row whose tag is not written can hand its place back to its base clip
    events_root = Path(a.events_root) if a.events_root and a.stage >= 2 else None
    events = []
    if events_root and (events_root / "metadata.jsonl").exists():
        if not (events_root / "quality.jsonl").exists():
            raise SystemExit(f"{events_root / 'quality.jsonl'} missing: run scripts/audit_quality.py --root {events_root}")
        quality.update({q["id"]: q for q in load_jsonl(events_root / "quality.jsonl")})
        events = with_wavs(load_jsonl(events_root / "metadata.jsonl"), events_root / a.events_wavs, "event clips")
        rows = combine_events(rows, events, quality, evidence=set(a.event_evidence.split(",")),
                              include_unconfirmed=a.include_unconfirmed, **tier_kw)
    elif a.stage >= 2:
        print("[warn] no event-extended clips (run ttsdata extend-events + verify-events, or pass --events-root)")
    rows = stage_filter(rows, soundtags, a.stage)
    funnel = tier_funnel(rows, quality, **tier_kw)
    tier = funnel[-1][1]
    root_of = {"dataset": os.path.relpath(wav_dir, out),
               "dataset_long": os.path.relpath(long_root / a.long_wavs, out) if long_root else None,
               "dataset_events": os.path.relpath(events_root / a.events_wavs, out) if events_root else None}

    # labels (stage 2): styles from the clip's (or its base's) subtitle label and, with --auto-styles, from the
    # automatic emotion label; tags from event clips and merged pauses
    auto: dict[str, list[str]] = {}
    if a.stage >= 2 and a.auto_styles:
        per_class = {k: float(v) for k, v in (kv.split("=") for kv in a.auto_conf.split(",") if kv.strip())}
        em_rows = [r for p in ([root] + ([long_root] if long_root else [])) for r in load_jsonl(p / "emotions.jsonl")]
        if not em_rows:
            raise SystemExit("--auto-styles needs emotions.jsonl: run ttsdata emotions first")
        auto = auto_styles(em_rows, a.auto_min_conf, per_class)
        print(f"auto styles: {len(auto)} clips above confidence ({a.auto_min_conf}{', ' + a.auto_conf if a.auto_conf else ''})")

    def tags_of(r):
        key = r.get("base") or r["id"]
        t = (soundtags or {}).get(key)
        extra = auto.get(key, [])
        if not extra:
            return t
        merged = dict(t) if t else {"style": [], "flags": [], "events": []}
        merged["style"] = sorted(set(merged.get("style", [])) | set(extra))
        return merged

    def labels_of(r):  # labels the row can carry in stage 2 (written only when frequent enough in the slice)
        t = tags_of(r) or {}
        labs = [f"style:{s}" for s in t.get("style", []) if s in STYLE_PREFIX]
        if r["root"] == "dataset_events" and r.get("tag") in EVENT_TAG:
            labs.append(f"tag:{r['tag']}")
        if r["root"] == "dataset_long":
            labs += [f"tag:{e['tag']}" for e in t.get("events", []) if (e.get("where") or "").startswith("gap:")
                     and e.get("char") is not None and e.get("who_match") is not False and e.get("tag") in EVENT_TAG]
        return labs

    # stage 2: labelled clips go in first (rarest label first, --label-quota each, at most --label-max-share of the
    # hours), so a random slice of a much bigger tier does not sample the rare tags away; the random fill then takes
    # unlabelled clips only, so the labelled share stays within the cap and each label keeps its quota
    first = []
    pool = tier
    if a.stage >= 2 and a.label_first:
        first = labels_first(tier, labels_of, a.label_max_share * a.max_hours, a.label_quota, a.seed)
        print(f"label-first: {len(first)} labelled clips ({sum(r['duration'] for r in first) / 3600:.2f} h) "
              "before the random fill")
        pool = first + [r for r in tier if not labels_of(r)]
    slice_rows = cap_hours(pool, a.max_hours, a.max_series_share, a.seed, taken=first)
    tier_labels = Counter(lab for r in tier for lab in labels_of(r)) if a.stage >= 2 else Counter()

    styles_seen = [s for r in slice_rows for s in (tags_of(r) or {}).get("style", [])]
    tags_seen = [r["tag"] for r in slice_rows if r["root"] == "dataset_events"]
    tags_seen += [e["tag"] for r in slice_rows if r["root"] == "dataset_long" for e in (tags_of(r) or {}).get("events", [])
                  if (e.get("where") or "").startswith("gap:") and e.get("who_match") is not False]
    styles_ok, events_ok = allowed_labels(styles_seen, tags_seen, a.min_style, a.min_tag) if a.stage >= 2 else (set(), set())
    if a.stage >= 2:  # an event clip whose tag is not written would carry an untagged sound: back to its base
        slice_rows = restore_bases(slice_rows, events_ok, units, quality, **tier_kw)
    # reference pairs are drawn after the slice is final, so no row points at a clip that left it
    train_rows = [r for r in slice_rows if r.get("split") != "val"]
    val_rows = [r for r in slice_rows if r.get("split") == "val"]
    pairs = {**pair_references(train_rows, a.ref_ratio, a.seed), **pair_references(val_rows, a.ref_ratio, a.seed)}
    by_id = {r["id"]: r for r in slice_rows}

    train, val = [], []
    n_labelled = 0
    for r in slice_rows:
        ref = by_id.get(pairs.get(r["id"]))
        text = r["normalized_text_tagged"] if r["root"] == "dataset_events" else r["normalized_text"]
        text = labelled_text(text, tags_of(r), styles_ok, events_ok) if a.stage >= 2 else text
        d = manifest_row({**r, "normalized_text": text}, ref, root_of[r["root"]], dataset_id=0,
                         ref_audio_root=root_of[ref["root"]] if ref else None)
        d["labelled"] = text != r["normalized_text"]
        n_labelled += d["labelled"]
        (val if r.get("split") == "val" else train).append(d)
    repeat_used = 1
    if a.stage >= 2 and a.label_repeat > 1:
        before = len(train)
        train = oversample(train, a.label_repeat, a.label_max_share, a.seed)
        n_lab = sum(1 for d in train if d["labelled"])
        repeat_used = round(n_lab / max(1, sum(1 for d in train[:before] if d["labelled"])), 2)
    for d in train + val:
        d.pop("labelled", None)

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
    n_events_slice = sum(1 for r in slice_rows if r["root"] == "dataset_events")
    covered = sum(len(r["members"]) for r in rows if r["root"] == "dataset_long")
    lines = [f"# VoxCPM manifest tier report ({a.wavs}/, stage {a.stage})", "",
             f"event-extended clips: {len(events)} loaded" if a.stage >= 2 else "stage 1: no labels, no event clips, strong-style rows left out",
             f"single clips: {len(singles)} ({hrs(singles):.2f} h); merged clips: {len(merged)} ({hrs(merged):.2f} h), "
             f"of which {sum(1 for r in rows if r['root'] == 'dataset_long')} pass the tier and replace {covered} single clips",
             f"sound flags: {'applied (' + str(len(soundtags)) + ' tagged clips)' if soundtags is not None else 'not available'}",
             "", f"units entering the funnel: {len(rows)} ({hrs(rows):.2f} h)", "",
             "| gate | clips | hours |", "|---|---|---|"]
    lines += [f"| {name} | {len(rs)} | {hrs(rs):.2f} |" for name, rs in funnel]
    lines += ["", f"slice (max {a.max_hours} h, series cap {a.max_series_share:.0%}): {len(slice_rows)} clips, {hrs(slice_rows):.2f} h, "
              f"of which {n_merged_slice} merged ({hrs([r for r in slice_rows if r['root'] == 'dataset_long']):.2f} h)"
              + (f" and {n_events_slice} event-extended" if a.stage >= 2 else ""), "",
              "| series | clips | hours | speakers |", "|---|---|---|---|"]
    per = defaultdict(list)
    for r in slice_rows:
        per[r["series"]].append(r)
    for s, rs in sorted(per.items(), key=lambda kv: -hrs(kv[1])):
        lines.append(f"| {s} | {len(rs)} | {hrs(rs):.2f} | {len({r['speaker_id'] for r in rs})} |")
    if a.stage >= 2:
        cs, ct = Counter(styles_seen), Counter(tags_seen)
        lines += ["", f"## Labels (stage 2; min_style {a.min_style}, min_tag {a.min_tag})", "",
                  "| label | kind | in tier | in slice | written |", "|---|---|---|---|---|"]
        lines += [f"| {s} | style | {tier_labels['style:' + s]} | {n} | {'yes' if s in styles_ok else 'no'} |"
                  for s, n in cs.most_common()]
        lines += [f"| {t} | event | {tier_labels['tag:' + t]} | {n} | {'yes' if t in events_ok else 'no'} |"
                  for t, n in ct.most_common()]
        lines += ["", f"- label-first: {len(first)} labelled clips ({hrs(first):.2f} h) taken before the random fill "
                  f"(quota {a.label_quota} per label, rarest first, at most {a.label_max_share:.0%} of {a.max_hours} h); "
                  "the random fill takes unlabelled clips only"
                  if a.label_first else "- label-first: off (labelled clips only as the random slice brings them)"]
        if a.auto_styles:
            lines += ["", f"- automatic styles (ttsdata emotions, confidence >= {a.auto_min_conf}"
                      f"{', ' + a.auto_conf if a.auto_conf else ''}): {sum(1 for r in slice_rows if auto.get(r.get('base') or r['id']))} clips in the slice"]
        lines += ["", f"- labelled rows: {n_labelled} before oversampling; repeat factor used {repeat_used} "
                  f"(requested {a.label_repeat}, cap {a.label_max_share:.0%} of train); event evidence accepted: {a.event_evidence}"
                  + (" + unconfirmed" if a.include_unconfirmed else "")]
    n_ref = sum(1 for d in train + val if "ref_audio" in d)
    lines += ["", f"- rows with ref_audio: {n_ref} ({n_ref / max(1, len(slice_rows)):.0%} of corpus rows)",
              f"- replay rows (dataset_id 1): {len(replay_rows)} ({len(replay_rows) / max(1, len(train)):.0%} of train)",
              f"- train {len(train)} / val {len(val)}; held-out eval sentences: {min(a.heldout, len(held))}", ""]
    (out / "tier_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
