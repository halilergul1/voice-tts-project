"""Event-extended clips: re-cut a clip so that the laugh / cry / sigh the subtitles place right before or after its
line is inside the audio, and tag it in the text ("[gülme] Hadi!").

Sources: the `events` of soundtags.jsonl (ttsdata soundtags) with where = before / after / lead on kept clips of
data/dataset/ and data/dataset_long/ (a merged clip is preferred over its members). The new span covers the event
cue plus event_pad, must stay under event_max_dur, must not overlap another clip's line, and must not be attributed
to another person by the subtitle. The span is cut from the raw episode audio, loudness-normalised and re-checked
with Whisper: the plain text must align (CER <= merge_max_cer) and the extended side may carry at most
event_max_bleed extra recognised tokens (a laugh is not words).

Output: data/dataset_events/ in the dataset layout; each row names its `base` clip (a manifest uses the extended
clip or its base, never both) and carries `verified: false` until a sound-event detector has confirmed the sound.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from .clipmerge import _cache_key, _cut_episode, _load_cache, _save_json, _verify, superseded
from .config import Config
from .labels import EXTENDABLE_TAGS, tag_text


def output_dir(cfg: Config, limit_episodes: int = 0) -> Path:
    return cfg.root / ("dataset_events_smoke" if limit_episodes else "dataset_events")


def extension(clip: dict, event: dict, pad: float = 0.2, max_dur: float = 15.0,
              neighbours: list | None = None) -> tuple[float, float, str] | None:
    """New (start, end, edge) covering the event, or None when it is not safe to extend.

    neighbours: (start, end[, id]) of other clips in the episode; the added audio must not overlap any of them."""
    if event.get("tag") not in EXTENDABLE_TAGS or event.get("who_match") is False:
        return None
    a, b = float(clip["orig_start"]), float(clip["orig_end"])
    where = event.get("where")
    if where in ("before", "lead"):
        start, end, edge = max(0.0, float(event["start"]) - pad), b, "start"
        added = (start, a)
    elif where == "after":
        start, end, edge = a, float(event["end"]) + pad, "end"
        added = (b, end)
    else:
        return None  # inside / gap: the position within the clip is not known precisely enough
    if end - start > max_dur or added[1] <= added[0]:
        return None
    for nb in neighbours or []:
        s, e = float(nb[0]), float(nb[1])
        if s < added[1] and e > added[0]:
            return None
    return round(start, 3), round(end, 3), edge


def candidates(bases: list[dict], tags: dict[str, dict], neighbours: dict[str, list] | None = None,
               pad: float = 0.2, max_dur: float = 15.0) -> list[dict]:
    """One candidate row per usable (clip, adjacent event). Rows carry the plain text for verification and the
    tagged text for training; a merged base's own members are not treated as neighbours."""
    out = []
    for base in bases:
        t = tags.get(base["id"])
        if not t:
            continue
        own = {base["id"]} | set(base.get("members") or [])
        nb = [n for n in (neighbours or {}).get(base["uid"], []) if not (len(n) > 2 and n[2] in own)]
        seen = set()
        for ev in t.get("events", []):
            ext = extension(base, ev, pad, max_dur, nb)
            if ext is None:
                continue
            start, end, edge = ext
            cid = f"{base['id']}_{ev['tag']}"
            if cid in seen:
                cid = f"{cid}_{edge}"
            seen.add(cid)
            row = {k: base[k] for k in ("uid", "source", "series", "episode", "speaker_id", "speaker_cluster_size",
                                        "speaker_hint", "split", "sample_rate") if k in base}
            row.update({
                "id": cid, "audio_file": f"wavs/{cid}.wav", "base": base["id"], "root": base.get("root", "dataset"),
                "orig_start": start, "orig_end": end, "duration": round(end - start, 3), "edge": edge, "tag": ev["tag"],
                "text": base["text"], "normalized_text": base["normalized_text"],  # plain, as every dataset tool expects
                "text_tagged": tag_text(base["text"], ev["tag"], edge),            # what the training manifest uses
                "normalized_text_tagged": tag_text(base["normalized_text"], ev["tag"], edge),
                "event": {k: ev.get(k) for k in ("tag", "where", "start", "end", "who", "who_match", "desc")},
                "flags": sorted(set(base.get("flags", [])) | {"event_extended"}), "verified": False, "detector": None,
            })
            out.append(row)
    return out


def accept(res: dict, edge: str, max_cer: float, max_bleed: int) -> str | None:
    """Rejection reason for a Whisper verdict on an extended clip, or None when it is kept."""
    if res.get("trim_start") is None:
        return "unaligned"
    if res["cer"] > max_cer:
        return "high_cer"
    bleed = res.get("bleed_before", 0) if edge == "start" else res.get("bleed_after", 0)
    if bleed > max_bleed:
        return "words_in_extension"
    return None


# ----------------------------------------------------------------------------------------------- driver
def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def run(cfg: Config, limit_episodes: int = 0, listen: int = 6) -> dict:
    from .asr import resolve_backend

    singles = _load_jsonl(cfg.dataset_dir / "metadata.jsonl")
    if not singles:
        raise SystemExit("data/dataset/metadata.jsonl missing: run `ttsdata build` first")
    merged = _load_jsonl(cfg.root / "dataset_long" / "metadata.jsonl")
    tags = {t["id"]: t for p in (cfg.dataset_dir, cfg.root / "dataset_long") for t in _load_jsonl(p / "soundtags.jsonl")}
    if not tags:
        raise SystemExit("soundtags.jsonl missing: run `ttsdata soundtags` first")
    if limit_episodes:
        keep = sorted({r["uid"] for r in singles})[:limit_episodes]
        singles = [r for r in singles if r["uid"] in keep]
        merged = [r for r in merged if r["uid"] in keep]
    sup = superseded(merged)
    bases = [{**r, "root": "dataset"} for r in singles if r["id"] not in sup] + [{**r, "root": "dataset_long"} for r in merged]
    neighbours: dict[str, list] = defaultdict(list)
    for r in singles:
        neighbours[r["uid"]].append((r["orig_start"], r["orig_end"], r["id"]))
    cands = candidates(bases, tags, neighbours, cfg.event_pad, cfg.event_max_dur)
    print(f"[events] {len(bases)} clips, {sum(len(t['events']) for t in tags.values())} events -> {len(cands)} candidates "
          f"({Counter(c['tag'] for c in cands).most_common()})")

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
    _verify(cfg, cands, wav_dir, cache, cache_file, model_tag)  # Whisper checks the plain words, never the tag

    kept, rejected = [], []
    reasons: Counter = Counter()
    for c in cands:
        res = cache[_cache_key(c, model_tag)]
        c.update(cer=res.get("cer"), asr_text=res.get("asr_text"), word_prob=res.get("word_prob"),
                 bleed=[res.get("bleed_before"), res.get("bleed_after")])
        why = accept(res, c["edge"], cfg.merge_max_cer, cfg.event_max_bleed)
        if why:
            reasons[why] += 1
            rejected.append({"id": c["id"], "reason": why, "text": c["text"], "asr_text": res.get("asr_text"),
                             "cer": res.get("cer"), "bleed": c["bleed"], "event": c["event"]})
        else:
            c["duration"] = round(sf.info(wav_dir / f"{c['id']}.wav").duration, 3)
            kept.append(c)
    keep_ids = {c["id"] for c in kept}
    for p in wav_dir.glob("*.wav"):
        if p.stem not in keep_ids:
            p.unlink()
    _save_json(index_file, {k: v for k, v in index.items() if k in keep_ids})

    def lj(r):  # the csv files are the training-facing view: tagged text
        return f"{r['id']}|{r['text_tagged']}|{r['normalized_text_tagged']}|{r['speaker_id']}\n"

    with (out / "metadata.jsonl").open("w", encoding="utf-8") as fj, (out / "metadata.csv").open("w", encoding="utf-8") as fa, \
         (out / "train.csv").open("w", encoding="utf-8") as ft, (out / "val.csv").open("w", encoding="utf-8") as fv:
        for r in kept:
            fj.write(json.dumps(r, ensure_ascii=False) + "\n")
            fa.write(lj(r))
            (fv if r.get("split") == "val" else ft).write(lj(r))
    with (out / "rejected.jsonl").open("w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    st = summarize(cands, kept, reasons, cfg)
    (out / "stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text(readme(st), encoding="utf-8")
    _write_listen(out, kept, listen)
    return st


def summarize(cands: list[dict], kept: list[dict], reasons: Counter, cfg: Config) -> dict:
    per_tag = {tag: {"candidates": sum(1 for c in cands if c["tag"] == tag), "kept": sum(1 for r in kept if r["tag"] == tag)}
               for tag in sorted({c["tag"] for c in cands})}
    return {
        "settings": {k: getattr(cfg, k) for k in ("event_pad", "event_max_dur", "event_max_bleed", "merge_max_cer")},
        "candidates": len(cands), "kept": len(kept), "rejected": dict(reasons.most_common()),
        "kept_hours": round(sum(r["duration"] for r in kept) / 3600, 2),
        "per_tag": per_tag,
        "by_edge": dict(Counter(r["edge"] for r in kept)),
        "by_root": dict(Counter(r["root"] for r in kept)),
        "who_match": dict(Counter(str(r["event"]["who_match"]) for r in kept)),
        "verified": sum(1 for r in kept if r["verified"]),
    }


def readme(st: dict) -> str:
    return "\n".join([
        "# Event-extended clips (`ttsdata extend-events`)", "",
        "Clips re-cut to include the laugh / cry / sigh the subtitles place right before or after the line, with the "
        "sound tagged in the text: `text_tagged` / `normalized_text_tagged` in `metadata.jsonl` and the csv files "
        "(`[gülme] Hadi!`); `text` / `normalized_text` stay plain so the audit and tier tools read them unchanged. "
        "Each row names its `base` clip; a manifest uses the extended clip or its base, never both. `verified` stays false until a sound-event detector confirms the sound is there; "
        "`event.who_match` is null when the subtitle does not say who made the sound.", "",
        f"- candidates: {st['candidates']}, kept after Whisper re-check: {st['kept']} ({st['kept_hours']} h), "
        f"rejected: {st['rejected']}",
        "- per tag (candidates / kept): " + ", ".join(f"{t} {v['candidates']}/{v['kept']}" for t, v in st["per_tag"].items()),
        f"- edge: {st['by_edge']}; base root: {st['by_root']}; who_match: {st['who_match']}",
        f"- settings: `{json.dumps(st['settings'])}`", "",
        "Listening sample: `listen_events/index.md`.", "",
    ])


def _write_listen(out: Path, kept: list[dict], n: int) -> None:
    import os
    import random

    d = out / "listen_events"
    d.mkdir(exist_ok=True)
    for p in d.iterdir():
        if p.is_symlink():
            p.unlink()
    rng = random.Random(1)
    idx = ["# Event-extended clips to listen to", "", "Is the tagged sound really there, at that edge, in this voice?", "",
           "| file | tag | edge | who | text |", "|---|---|---|---|---|"]
    k = 0
    for tag in sorted({r["tag"] for r in kept}):
        rows = [r for r in kept if r["tag"] == tag]
        for r in rng.sample(rows, min(n, len(rows))):
            name = f"{k:03d}_{tag}_{r['edge']}_{r['id']}.wav"
            os.symlink(os.path.relpath(out / "wavs" / f"{r['id']}.wav", d), d / name)
            idx.append(f"| `{name}` | {tag} | {r['edge']} | {r['event']['who'] or '-'} | {r['text_tagged']} |")
            k += 1
    (d / "index.md").write_text("\n".join(idx) + "\n", encoding="utf-8")
