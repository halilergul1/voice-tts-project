"""Merge adjacent same-speaker clips into longer utterances (VoxCPM wants 3–30 s; two thirds of our clips are < 3 s).

A chain joins clips of one episode that are consecutive subtitle segments (no dropped cue in between), close in time
(gap <= merge_max_gap), spoken by the same voice (ECAPA cosine to the previous clip and to the chain centroid
>= merge_min_cos), in the same train/val split, with compatible speaker labels, and no longer than merge_max_dur.
A link across two speaker clusters needs merge_cross_cluster_cos, a link after a line ending in "?" needs
merge_after_question_cos (both higher than merge_min_cos: turn changes hide there). Each chain's span is cut again from the raw episode audio (so the pause between the lines is real audio), loudness
normalised, re-transcribed with Whisper and kept only when its CER against the joined subtitle text is
<= merge_max_cer (this rejects chains whose gap holds the audio-description narrator or another speaker).

Output: data/dataset_long/ in the dataset layout (metadata.jsonl, metadata.csv / train.csv / val.csv with speaker_id,
wavs/, rejected.jsonl, stats.json, README.md), so audit_quality.py, separate_vocals.py and the manifest builder run on
it unchanged. Each row lists its "members"; a training manifest uses either the merged clip or its members, never both.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from .config import Config
from .soundtags import hint_style, tr_lower


def long_dir(cfg: Config) -> Path:
    return cfg.root / "dataset_long"


def output_dir(cfg: Config, limit_episodes: int = 0) -> Path:
    """Smoke runs (--limit-episodes) write next to the real output, never over it."""
    return cfg.root / ("dataset_long_smoke" if limit_episodes else "dataset_long")


def seg_index(clip_id: str) -> int:
    return int(clip_id.rsplit("_", 1)[1])


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _hint_label(hint: str | None) -> str | None:
    """The speaker part of a subtitle label, whole: "Nisan ses" -> "nisan", "Erkek 2" -> "erkek 2",
    "Ömer ağlayarak" -> "ömer" (the manner is not part of the name)."""
    name = hint_style(hint)[0]
    if not name:
        return None
    low = re.sub(r"\s+(?:dış ses|iç ses|ses)$", "", tr_lower(name)).strip()
    return re.sub(r"\s+", " ", low) or None


def _compatible(chain: list[dict], nxt: dict) -> bool:
    labels = {_hint_label(r.get("speaker_hint")) for r in chain} - {None}
    other = _hint_label(nxt.get("speaker_hint"))
    return other is None or not labels or labels == {other}


def _link_threshold(prev: dict, nxt: dict, min_cos: float, cross_cluster_cos: float, after_question_cos: float) -> float:
    """Voice similarity a link needs: higher across speaker clusters and after a question (turn changes hide there)."""
    t = min_cos
    if prev.get("speaker_id") != nxt.get("speaker_id"):
        t = max(t, cross_cluster_cos)
    if prev["text"].rstrip().endswith("?"):
        t = max(t, after_question_cos)
    return t


def build_chains(rows: list[dict], vectors: dict[str, np.ndarray], max_gap: float, min_cos: float, max_dur: float,
                 min_gap: float = -0.5, cross_cluster_cos: float | None = None,
                 after_question_cos: float | None = None) -> list[list[dict]]:
    """Greedy left-to-right chains (length >= 2) of mergeable clips, per episode."""
    cross_cluster_cos = min_cos if cross_cluster_cos is None else cross_cluster_cos
    after_question_cos = min_cos if after_question_cos is None else after_question_cos
    by_uid: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_uid[r["uid"]].append(r)
    chains: list[list[dict]] = []
    for ep in by_uid.values():
        ep = sorted(ep, key=lambda r: (r["orig_start"], seg_index(r["id"])))
        cur: list[dict] = []
        centroid = None
        for r in ep:
            v = vectors.get(r["id"])
            ok = False
            if cur and v is not None:
                p = cur[-1]
                gap = r["orig_start"] - p["orig_end"]
                need = _link_threshold(p, r, min_cos, cross_cluster_cos, after_question_cos)
                ok = (seg_index(r["id"]) == seg_index(p["id"]) + 1
                      and min_gap <= gap <= max_gap
                      and r.get("split") == p.get("split")
                      and r["orig_end"] - cur[0]["orig_start"] <= max_dur
                      and float(_unit(v) @ _unit(vectors[p["id"]])) >= need
                      and float(_unit(v) @ _unit(centroid)) >= min_cos
                      and _compatible(cur, r))
            if ok:
                cur.append(r)
                centroid = centroid + _unit(v)
            else:
                if len(cur) >= 2:
                    chains.append(cur)
                cur, centroid = ([r], _unit(v)) if v is not None else ([], None)
        if len(cur) >= 2:
            chains.append(cur)
    return chains


def join_texts(texts: list[str]) -> tuple[str, list[int]]:
    """Members' texts joined by a space, plus the character offset where each member's text ends."""
    out, ends = "", []
    for t in texts:
        t = t.strip()
        out = f"{out} {t}" if out else t
        ends.append(len(out))
    return out, ends


def merged_row(chain: list[dict]) -> dict:
    first, last = chain[0], chain[-1]
    mid = f"{first['id']}_x{len(chain)}"
    text, ends = join_texts([r["text"] for r in chain])
    norm, _ = join_texts([r["normalized_text"] for r in chain])
    by_spk: Counter = Counter()
    size_of = {}
    for r in chain:
        by_spk[r["speaker_id"]] += r["orig_end"] - r["orig_start"]
        size_of[r["speaker_id"]] = r.get("speaker_cluster_size")
    spk = by_spk.most_common(1)[0][0]
    hint = next((n for n in (hint_style(r.get("speaker_hint"))[0] for r in chain) if n), None)  # name part only
    flags = {"merged"}
    flags |= {f for f in first.get("flags", []) if f in ("fragment_start", "edge_unmatched")}
    flags |= {f for f in last.get("flags", []) if f in ("fragment_end", "edge_unmatched")}
    return {
        "id": mid, "audio_file": f"wavs/{mid}.wav", "text": text, "normalized_text": norm,
        "duration": round(last["orig_end"] - first["orig_start"], 3), "sample_rate": first.get("sample_rate"),
        "source": first["source"], "series": first["series"], "episode": first.get("episode"), "uid": first["uid"],
        "speaker_hint": hint, "speaker_id": spk, "speaker_cluster_size": size_of[spk],
        "speaker_ids": sorted(by_spk), "orig_start": first["orig_start"], "orig_end": last["orig_end"],
        "members": [r["id"] for r in chain], "member_spans": [[r["orig_start"], r["orig_end"]] for r in chain],
        "member_hints": [r.get("speaker_hint") for r in chain],
        "member_text_ends": ends, "gaps": [round(b["orig_start"] - a["orig_end"], 3) for a, b in zip(chain, chain[1:])],
        "flags": sorted(flags), "split": first.get("split", "train"),
    }


def listen_picks(kept: list[dict], weakest: dict[str, float], n: int, seed: int = 0) -> list[tuple[str, str]]:
    """Merged clips to listen to: the n with the weakest voice link between members, plus n random others."""
    import random

    weak = sorted((r["id"] for r in kept), key=lambda i: weakest.get(i, 1.0))[:n]
    rest = sorted(r["id"] for r in kept if r["id"] not in set(weak))
    rand = random.Random(seed).sample(rest, min(n, len(rest)))
    return [("weakest_link", i) for i in weak] + [("random", i) for i in rand]


def superseded(merged: list[dict]) -> set[str]:
    """Ids of single clips covered by kept merged clips (a manifest uses one or the other)."""
    return {m for row in merged for m in row["members"]}


# ----------------------------------------------------------------------------------------------- driver
def _cache_key(row: dict, model_tag: str) -> str:
    """Whisper verdicts are only reused for the same span, the same text and the same recogniser / cut settings."""
    h = hashlib.sha1(row["text"].encode("utf-8")).hexdigest()[:10]
    return f"{row['id']}|{row['orig_start']:.3f}|{row['orig_end']:.3f}|{h}|{model_tag}"


def _cut_key(row: dict, cfg: Config) -> str:
    return f"{row['orig_start']:.3f}|{row['orig_end']:.3f}|{cfg.sample_rate}|{cfg.target_lufs}|{cfg.peak_ceiling}"


def _save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    os.replace(tmp, path)


def _load_cache(path: Path) -> dict[str, dict]:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a killed run may leave half a line
                out[r["key"]] = r
    return out


def _cut_episode(cfg: Config, uid: str, source: str, cands: list[dict], wav_dir: Path, index: dict[str, str]) -> None:
    """Cut every candidate whose wav is missing or was cut from another span / with other settings.

    index maps clip id -> _cut_key of the audio on disk; wavs are written to a temp name and renamed, so a killed run
    never leaves a partial wav that looks finished."""
    from .segment import decode_audio
    from .separation import loudness_normalize

    todo = [c for c in cands if index.get(c["id"]) != _cut_key(c, cfg) or not (wav_dir / f"{c['id']}.wav").exists()]
    if not todo:
        return
    raw = next((cfg.raw_dir / source / uid).glob("audio.*"), None)
    if raw is None:
        raise SystemExit(f"raw audio for {uid} not found under {cfg.raw_dir / source / uid}")
    sr = cfg.sample_rate
    audio = decode_audio(raw, sr)
    for c in todo:
        y = audio[int(c["orig_start"] * sr):int(c["orig_end"] * sr)]
        y = loudness_normalize(y, sr, cfg.target_lufs, cfg.peak_ceiling)
        tmp = wav_dir / f"{c['id']}.tmp.wav"
        sf.write(tmp, y, sr, subtype="PCM_16")
        os.replace(tmp, wav_dir / f"{c['id']}.wav")
        index[c["id"]] = _cut_key(c, cfg)


def _verify(cfg: Config, cands: list[dict], wav_dir: Path, cache: dict[str, dict], cache_file: Path,
            model_tag: str) -> None:
    from . import asr

    todo = [{"id": c["id"], "wav": str(wav_dir / f"{c['id']}.wav"), "window_dur": c["duration"], "text": c["text"],
             "key": _cache_key(c, model_tag)} for c in cands if _cache_key(c, model_tag) not in cache]
    if not todo:
        return
    packs = asr.pack_records(todo, cfg.pack_seconds, cfg.pack_gap, cfg.seed) if cfg.pack_seconds > 0 else [[r] for r in todo]
    with cache_file.open("a", encoding="utf-8") as f, tqdm(total=len(todo), desc="whisper", unit="clip") as bar:
        for pack in packs:
            if cfg.pack_seconds > 0:
                results = asr.verify_pack(pack, cfg)
            else:
                results = [{**asr.evaluate(pack[0], asr.transcribe_words(pack[0]["wav"], cfg), cfg), "id": pack[0]["id"]}]
            for rec, res in zip(pack, results):
                res = {k: v for k, v in res.items() if k not in ("words",)}
                res["key"] = rec["key"]
                cache[rec["key"]] = res
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            bar.update(len(pack))


def _write_listen(out: Path, kept: list[dict], weakest: dict[str, float], n: int) -> None:
    import os

    d = out / "listen_merges"  # audit_quality.py owns listen/
    d.mkdir(exist_ok=True)
    for p in d.iterdir():
        if p.is_symlink():
            p.unlink()
    row = {r["id"]: r for r in kept}
    idx = ["# Merged clips to listen to", "", "Same voice all the way through? Text matches?", "",
           "| file | group | weakest voice link (cos) | clusters | text |", "|---|---|---|---|---|"]
    for k, (group, cid) in enumerate(listen_picks(kept, weakest, n)):
        name = f"{k:03d}_{group}_{cid}.wav"
        os.symlink(os.path.relpath(out / "wavs" / f"{cid}.wav", d), d / name)
        r = row[cid]
        idx.append(f"| `{name}` | {group} | {weakest[cid]:.2f} | {len(r['speaker_ids'])} | {r['text']} |")
    (d / "index.md").write_text("\n".join(idx) + "\n", encoding="utf-8")


def run(cfg: Config, limit_episodes: int = 0, listen: int = 10) -> dict:
    from .speakers import embed_all, load_rows

    rows = load_rows(cfg)
    if not rows or "speaker_id" not in rows[0]:
        raise SystemExit("metadata.jsonl has no speaker_id: run `ttsdata speakers` first")
    if limit_episodes:
        keep = sorted({r["uid"] for r in rows})[:limit_episodes]
        rows = [r for r in rows if r["uid"] in keep]
    vectors = embed_all(cfg, rows)
    cos_of = lambda x, y: float(_unit(vectors[x]) @ _unit(vectors[y]))  # noqa: E731
    chains = build_chains(rows, vectors, cfg.merge_max_gap, cfg.merge_min_cos, cfg.merge_max_dur,
                          cross_cluster_cos=cfg.merge_cross_cluster_cos, after_question_cos=cfg.merge_after_question_cos)
    cands = [merged_row(c) for c in chains]
    print(f"[merge] {len(rows)} clips -> {len(chains)} chains ({sum(len(c) for c in chains)} clips, "
          f"{sum(c['duration'] for c in cands) / 3600:.2f} h)")

    from .asr import resolve_backend

    out = output_dir(cfg, limit_episodes)
    wav_dir = out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    index_file = out / "cut_index.json"
    index: dict[str, str] = json.loads(index_file.read_text(encoding="utf-8")) if index_file.exists() else {}
    by_uid: dict[str, list[dict]] = defaultdict(list)
    for c in cands:
        by_uid[c["uid"]].append(c)
    for uid, group in tqdm(sorted(by_uid.items()), desc="cutting", unit="episode"):
        _cut_episode(cfg, uid, group[0]["source"], group, wav_dir, index)
        _save_json(index_file, index)
    backend, model = resolve_backend(cfg)
    model_tag = f"{backend}:{model}|{cfg.sample_rate}|{cfg.target_lufs}"
    cache_file = out / "asr_cache.jsonl"
    cache = _load_cache(cache_file)
    _verify(cfg, cands, wav_dir, cache, cache_file, model_tag)

    kept, rejected = [], []
    reasons: Counter = Counter()
    for c in cands:
        res = cache[_cache_key(c, model_tag)]
        c.update(cer=res.get("cer"), asr_text=res.get("span_text"), word_prob=res.get("word_prob"),
                 bleed=[res.get("bleed_before"), res.get("bleed_after")])
        why = None
        if res.get("trim_start") is None:
            why = "unaligned"
        elif res["cer"] > cfg.merge_max_cer:
            why = "high_cer"
        if why:
            reasons[why] += 1
            rejected.append({"id": c["id"], "reason": why, "text": c["text"], "asr_text": res.get("asr_text"),
                             "cer": res.get("cer"), "members": c["members"]})
        else:
            c["duration"] = round(sf.info(wav_dir / f"{c['id']}.wav").duration, 3)
            kept.append(c)
    keep_ids = {c["id"] for c in kept}
    for p in wav_dir.glob("*.wav"):
        if p.stem not in keep_ids:
            p.unlink()  # rejected, stale (earlier build) or a leftover temp file; the Whisper cache keeps verdicts
    _save_json(index_file, {k: v for k, v in index.items() if k in keep_ids})

    def lj(r):
        return f"{r['id']}|{r['text']}|{r['normalized_text']}|{r['speaker_id']}\n"

    with (out / "metadata.jsonl").open("w", encoding="utf-8") as fj, (out / "metadata.csv").open("w", encoding="utf-8") as fa, \
         (out / "train.csv").open("w", encoding="utf-8") as ft, (out / "val.csv").open("w", encoding="utf-8") as fv:
        for r in kept:
            fj.write(json.dumps(r, ensure_ascii=False) + "\n")
            fa.write(lj(r))
            (fv if r["split"] == "val" else ft).write(lj(r))
    with (out / "rejected.jsonl").open("w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    st = summarize(rows, cands, kept, reasons, cfg)
    (out / "stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    weakest = {r["id"]: min(cos_of(a, b) for a, b in zip(r["members"], r["members"][1:])) for r in kept}
    _write_listen(out, kept, weakest, listen)
    (out / "README.md").write_text(readme(st), encoding="utf-8")
    return st


def summarize(rows: list[dict], cands: list[dict], kept: list[dict], reasons: Counter, cfg: Config,
              tier_cer: float = 0.05) -> dict:
    sup = superseded(kept)

    def hours(rs, min_dur=0.0):
        return round(sum(r["duration"] for r in rs if r["duration"] >= min_dur) / 3600, 2)

    singles_left = [r for r in rows if r["id"] not in sup]

    def tier(rs):  # the manifest's own gates on length and CER (other gates apply to both alike)
        return [r for r in rs if r["duration"] >= 3.0 and r.get("cer") is not None and r["cer"] <= tier_cer]

    tier_merged = tier(kept)
    tier_sup = superseded(tier_merged)  # members of merges that fail the tier stay usable as singles
    per_series = defaultdict(lambda: {"merged": 0, "members": 0})
    for r in kept:
        per_series[r["series"]]["merged"] += 1
        per_series[r["series"]]["members"] += len(r["members"])
    return {
        "settings": {k: getattr(cfg, k) for k in ("merge_max_gap", "merge_min_cos", "merge_cross_cluster_cos",
                                                  "merge_after_question_cos", "merge_max_dur", "merge_max_cer")},
        "single_clips": len(rows), "single_hours": hours(rows),
        "chains": len(cands), "kept": len(kept), "rejected": dict(reasons.most_common()),
        "kept_hours": hours(kept), "members_superseded": len(sup),
        "members_per_merge": round(float(np.mean([len(r["members"]) for r in kept])), 2) if kept else 0.0,
        "mixed_cluster_merges": sum(1 for r in kept if len(r["speaker_ids"]) > 1),
        "tier_hours_singles_only": round(sum(r["duration"] for r in tier(rows)) / 3600, 3),
        "tier_hours_with_merges": round(sum(r["duration"] for r in tier(tier_merged)
                                            + tier([r for r in rows if r["id"] not in tier_sup])) / 3600, 3),
        "hours_ge_3s_singles_only": hours(rows, 3.0),
        "hours_ge_3s_with_merges": round(hours(singles_left, 3.0) + hours(kept, 3.0), 2),
        "clips_ge_3s_singles_only": sum(1 for r in rows if r["duration"] >= 3.0),
        "clips_ge_3s_with_merges": sum(1 for r in singles_left if r["duration"] >= 3.0) + sum(1 for r in kept if r["duration"] >= 3.0),
        "per_series": dict(sorted(per_series.items())),
    }


def readme(st: dict) -> str:
    return "\n".join([
        "# Merged utterances (`ttsdata merge-clips`)", "",
        "Adjacent subtitle lines of one speaker joined into longer clips and cut again from the raw episode audio.",
        "Each row of `metadata.jsonl` lists its `members` (single clips in `data/dataset/`); a training manifest uses "
        "the merged clip or its members, never both. Same columns as the dataset, plus `members`, `member_spans`, "
        "`member_text_ends`, `gaps`, `speaker_ids`.", "",
        f"- chains found: {st['chains']}, kept after Whisper re-check: {st['kept']} ({st['kept_hours']} h), "
        f"rejected: {st['rejected']}",
        f"- hours in clips >= 3 s: {st['hours_ge_3s_singles_only']} h from single clips alone, "
        f"{st['hours_ge_3s_with_merges']} h when merged clips replace their members",
        f"- at the manifest gate (>= 3 s and CER <= 0.05): {st['tier_hours_singles_only']} h singles only, "
        f"{st['tier_hours_with_merges']} h with merges",
        "- listening sample: `listen_merges/index.md` (weakest voice links first, then random)",
        f"- settings: `{json.dumps(st['settings'])}`", "",
    ])
