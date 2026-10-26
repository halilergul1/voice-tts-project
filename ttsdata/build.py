"""Assemble the final dataset: filter, loudness-normalise, write LJSpeech-style metadata."""
from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from tqdm import tqdm

import re

from .asr import speech_cover, trim_bounds
from .config import Config
from .textnorm import basic_clean, charset_ok, normalize


def load_records(cfg: Config) -> list[dict]:
    recs = []
    for seg_file in sorted(cfg.seg_dir.glob("*/*/segments.jsonl")):
        asr = {}
        asr_file = seg_file.with_name("asr.jsonl")
        if asr_file.exists():
            for line in asr_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    a = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a worker may be mid-write on the last line
                asr[a["id"]] = a
        for line in seg_file.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            r.update({k: v for k, v in asr.get(r["id"], {}).items() if k != "id"})
            recs.append(r)
    return recs


CREDITS_RE = re.compile(r"betimleme|altyazı|işaret dili|yapım tarafından|izlediğiniz için", re.IGNORECASE)


def reject_reason(r: dict, cfg: Config) -> str | None:
    if "multi_speaker" in r["flags"]:
        return "multi_speaker"
    if CREDITS_RE.search(r["text"]):
        return "credits"  # accessibility credit lines, read by the narrator over the theme music
    if "no_speech" in r["flags"] or "wav" not in r:
        return "no_speech"
    text = basic_clean(r["text"])
    if len(text) < cfg.min_chars:
        return "text_too_short"
    cue_dur = r["cue_end"] - r["cue_start"]
    if cue_dur < cfg.min_dur - 0.4:
        return "too_short"
    if cue_dur > cfg.max_dur + 0.5:
        return "too_long"
    if speech_cover(r) < cfg.min_speech_ratio:
        return "low_speech_ratio"
    norm = normalize(text)
    if not charset_ok(norm):
        return "bad_charset"
    if "cer" not in r:
        return "asr_missing" if cfg.require_asr else None
    if r.get("error"):
        return "asr_error"
    if r.get("trim_start") is None:
        return "unaligned"
    if "vad_no_speech" in r["flags"]:
        return "no_speech"
    if r.get("no_speech_prob", 0) > cfg.max_no_speech_prob:
        return "asr_no_speech"
    if r["cer"] > cfg.max_cer:
        return "high_cer"
    if cfg.min_word_prob and r.get("word_prob", 1.0) < cfg.min_word_prob:
        return "low_word_prob"
    dur = r["trim_end"] - r["trim_start"]
    if dur < cfg.min_dur:
        return "too_short"
    if dur > cfg.max_dur:
        return "too_long"
    cps = len(norm) / dur
    if cps < cfg.min_cps:
        return "cps_too_low"
    if cps > cfg.max_cps:
        return "cps_too_high"
    return None


def snap_edges(regions: list[tuple[float, float]], dur: float, cfg: Config) -> tuple[float, float]:
    """Given VAD speech regions (seconds, clip-relative) return new (start, end) inside [0, dur] with the
    leading / trailing non-speech capped at cfg.lead_silence_max / cfg.trail_silence_max."""
    if not regions:
        return 0.0, dur
    a, b = 0.0, dur
    lead = regions[0][0]
    if lead > cfg.lead_silence_max:
        a = max(0.0, lead - cfg.snap_margin)
    trail = dur - regions[-1][1]
    if trail > cfg.trail_silence_max:
        b = min(dur, regions[-1][1] + cfg.snap_margin * 2)
    return a, b


def snap_record(r: dict, cfg: Config) -> None:
    """Tighten r['trim_start'] / r['trim_end'] on the window wav using VAD; marks r['snapped'] seconds removed."""
    from .segment import speech_regions

    y, sr = sf.read(r["wav"], dtype="float32")
    t0, t1 = r["trim_start"], r["trim_end"]
    x = y[int(t0 * sr):int(t1 * sr)]
    if len(x) < sr // 5:
        return
    regions = speech_regions(x, sr, cfg.vad_threshold)
    if not regions:
        r["flags"] = list(r.get("flags", [])) + ["vad_no_speech"]
        return
    a, b = snap_edges(regions, len(x) / sr, cfg)
    r["snapped"] = round((a) + (len(x) / sr - b), 3)
    r["trim_start"], r["trim_end"] = round(t0 + a, 3), round(t0 + b, 3)


def _loudness_normalize(y: np.ndarray, sr: int, target: float, ceiling: float) -> np.ndarray:
    meter = pyln.Meter(sr)
    try:
        loud = meter.integrated_loudness(y.astype(np.float64))
    except ValueError:
        loud = float("nan")
    if np.isfinite(loud):
        y = pyln.normalize.loudness(y, loud, target)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > ceiling:
        y = y * (ceiling / peak)
    return y.astype(np.float32)


def build(cfg: Config) -> dict:
    recs = load_records(cfg)
    for r in tqdm(recs, desc="edges", unit="seg", mininterval=5.0):
        # margins are a build-time choice: recompute edges from the stored word span, then snap with VAD
        if r.get("span_start") is not None:
            t0, t1 = trim_bounds(r["span_start"], r["span_end"], r["prev_end"], r["next_start"], r["first_ok"],
                                 r["last_ok"], r["window_dur"], cfg)
            r["trim_start"], r["trim_end"] = round(t0, 3), round(t1, 3)
        if r.get("trim_start") is not None and "wav" in r and r["trim_end"] > r["trim_start"]:
            snap_record(r, cfg)
    out = cfg.dataset_dir
    wavs = out / "wavs"
    if out.exists():
        shutil.rmtree(out)
    wavs.mkdir(parents=True)

    kept, rejected = [], []
    reasons = Counter()
    for r in recs:
        why = reject_reason(r, cfg)
        if why:
            reasons[why] += 1
            rejected.append({"id": r["id"], "reason": why, "text": r["text"], "duration": r.get("duration"),
                             "cer": r.get("cer"), "asr_text": r.get("asr_text")})
        else:
            kept.append(r)

    rows = []
    for r in tqdm(kept, desc="writing wavs", unit="seg"):
        y, sr = sf.read(r["wav"], dtype="float32")
        y = y[int(r["trim_start"] * sr):int(r["trim_end"] * sr)]
        y = _loudness_normalize(y, sr, cfg.target_lufs, cfg.peak_ceiling)
        dst = wavs / f"{r['id']}.wav"
        sf.write(dst, y, sr, subtype="PCM_16")
        text = basic_clean(r["text"])
        rows.append({
            "id": r["id"], "audio_file": f"wavs/{r['id']}.wav", "text": text, "normalized_text": normalize(text),
            "duration": round(len(y) / sr, 3), "sample_rate": sr, "source": r["source"], "series": r["series"],
            "episode": r["episode"], "uid": r["uid"], "speaker_hint": r.get("speaker_hint"),
            "orig_start": round(r["window_start"] + r["trim_start"], 3),
            "orig_end": round(r["window_start"] + r["trim_end"], 3),
            "cer": r.get("cer"), "asr_text": r.get("span_text"), "word_prob": r.get("word_prob"),
            "flags": [f for f in r["flags"] if f != "merged"] + (["edge_unmatched"] if r.get("edge_unmatched") else []),
        })

    rng = random.Random(cfg.seed)
    ids = [row["id"] for row in rows]
    rng.shuffle(ids)
    n_val = max(50, int(len(ids) * cfg.val_fraction)) if len(ids) > 100 else max(1, len(ids) // 10)
    val_ids = set(ids[:n_val])
    for row in rows:
        row["split"] = "val" if row["id"] in val_ids else "train"

    def lj_line(row):
        return f"{row['id']}|{row['text']}|{row['normalized_text']}\n"

    with (out / "metadata.csv").open("w", encoding="utf-8") as f_all, \
         (out / "train.csv").open("w", encoding="utf-8") as f_tr, \
         (out / "val.csv").open("w", encoding="utf-8") as f_va, \
         (out / "metadata.jsonl").open("w", encoding="utf-8") as f_js:
        for row in rows:
            f_all.write(lj_line(row))
            (f_va if row["split"] == "val" else f_tr).write(lj_line(row))
            f_js.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "rejected.jsonl").open("w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = summarize(rows, reasons, len(recs), cfg)
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text(readme(stats, cfg), encoding="utf-8")
    return stats


def summarize(rows: list[dict], reasons: Counter, n_total: int, cfg: Config) -> dict:
    durs = np.array([r["duration"] for r in rows]) if rows else np.zeros(0)
    per_series = defaultdict(lambda: {"utterances": 0, "seconds": 0.0, "episodes": set()})
    for r in rows:
        s = per_series[f"{r['source']}: {r['series']}"]
        s["utterances"] += 1
        s["seconds"] += r["duration"]
        s["episodes"].add(r["uid"])
    series = {k: {"utterances": v["utterances"], "hours": round(v["seconds"] / 3600, 2), "episodes": len(v["episodes"])}
              for k, v in sorted(per_series.items())}
    cers = [r["cer"] for r in rows if r.get("cer") is not None]
    return {
        "utterances": len(rows),
        "hours": round(float(durs.sum()) / 3600, 2) if len(durs) else 0.0,
        "train": sum(1 for r in rows if r["split"] == "train"),
        "val": sum(1 for r in rows if r["split"] == "val"),
        "duration_sec": {"mean": round(float(durs.mean()), 2), "median": round(float(np.median(durs)), 2),
                         "p5": round(float(np.percentile(durs, 5)), 2), "p95": round(float(np.percentile(durs, 95)), 2),
                         "min": round(float(durs.min()), 2), "max": round(float(durs.max()), 2)} if len(durs) else {},
        "cer": {"mean": round(float(np.mean(cers)), 4), "median": round(float(np.median(cers)), 4)} if cers else None,
        "sample_rate": cfg.sample_rate,
        "segments_considered": n_total,
        "rejected": dict(reasons.most_common()),
        "per_series": series,
        "filters": {k: getattr(cfg, k) for k in ("min_dur", "max_dur", "min_speech_ratio", "min_cps", "max_cps",
                                                   "max_cer", "max_no_speech_prob", "min_word_prob", "target_lufs",
                                                   "trim_margin_before", "trim_margin_after", "lead_silence_max",
                                                   "trail_silence_max", "snap_margin")},
    }


def readme(st: dict, cfg: Config) -> str:
    lines = [
        "# Turkish TTS corpus from accessible (Engelsiz) TV broadcasts",
        "",
        f"- utterances: **{st['utterances']}** ({st['hours']} h), train {st['train']} / val {st['val']}",
        f"- audio: mono {st['sample_rate']} Hz 16-bit PCM WAV, loudness-normalised to {cfg.target_lufs} LUFS",
        f"- utterance duration: median {st['duration_sec'].get('median')} s, p5 {st['duration_sec'].get('p5')} s, p95 {st['duration_sec'].get('p95')} s",
        f"- subtitle vs Whisper CER: mean {st['cer']['mean'] if st['cer'] else 'n/a'}, median {st['cer']['median'] if st['cer'] else 'n/a'}",
        "",
        "## Files",
        "",
        "| file | content |",
        "|---|---|",
        "| `wavs/<id>.wav` | one utterance per file |",
        "| `metadata.csv` | LJSpeech format: `id\\|text\\|normalized_text` (no header); after `ttsdata speakers` a 4th column `\\|speaker_id` |",
        "| `train.csv`, `val.csv` | same format, seeded random split |",
        "| `metadata.jsonl` | one JSON object per utterance with duration, source, series, episode, speaker_hint, speaker_id, original timestamps, Whisper transcript and CER |",
        "| `rejected.jsonl` | every candidate that was filtered out, with the reason |",
        "| `stats.json` | the numbers below |",
        "",
        "## Text fields",
        "",
        "- `text`: subtitle text with sound descriptions, speaker labels and dialogue dashes removed; original casing and punctuation.",
        "- `normalized_text`: `text` with quotes/dashes unified and digits expanded to Turkish words (num2words).",
        "- `speaker_hint`: the off-screen speaker label from the subtitle when present (e.g. `Usta`), otherwise null. Speaker identity is NOT labelled otherwise; the corpus is multi-speaker.",
        "- `speaker_id` / `speaker_cluster_size` (added by `ttsdata speakers`): unsupervised speaker id `<series>_spkNN` from clustered ECAPA-TDNN speaker embeddings (per episode, merged across the series; `spk00` has the most clips) and the number of clips sharing it. Ids are unique within a series only; a voice may be split over several ids and a small cluster may mix voices, so filter on `speaker_cluster_size` for speaker-conditioned training.",
        "- `flags`: `fragment_start` / `fragment_end` (sentence continues in a neighbouring cue, joined with `...`), `edge_unmatched` (a subtitle word at the clip edge had no Whisper word, so the clip keeps the audio gap there; slightly looser edges).",
        "",
        "## Rejection reasons",
        "",
        "| reason | count |",
        "|---|---|",
    ]
    for k, v in st["rejected"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Per series", "", "| series | episodes | utterances | hours |", "|---|---|---|---|"]
    for k, v in st["per_series"].items():
        lines.append(f"| {k} | {v['episodes']} | {v['utterances']} | {v['hours']} |")
    if st.get("speakers"):
        sp = st["speakers"]
        lines += ["", "## Speakers", "",
                  f"`ttsdata speakers`: {sp['model']} embeddings, average-linkage cosine threshold {sp['threshold']} within an episode, "
                  f"centroid merge threshold {sp['merge_threshold']} across a series. **{sp['total']}** speaker ids. "
                  f"Against the {sp['hinted_clips']} clips with a named `speaker_hint`: purity {sp['hint_purity']}, completeness {sp['hint_completeness']}.",
                  "", "| series | speakers | with >=10 clips | clips covered by them | largest share | hinted clips | purity | completeness |", "|---|---|---|---|---|---|---|---|"]
        for k, v in sp["per_series"].items():
            na = lambda x: "n/a" if x is None else x  # noqa: E731  (series without named speaker_hint clips)
            lines.append(f"| {k} | {v['speakers']} | {v['speakers_ge10']} | {v['coverage_ge10']} | {v['largest_share']} | {v['n']} | {na(v['purity'])} | {na(v['completeness'])} |")
    lines += ["", "## Filters", "", "```json", json.dumps(st["filters"], indent=2), "```", ""]
    return "\n".join(lines)
