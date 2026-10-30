"""Build VoxCPM fine-tuning manifests from the dataset: strict tier selection, same-speaker reference pairing,
row construction and validation. Pure functions; scripts/make_voxcpm_manifest.py does the I/O.
"""
from __future__ import annotations

import random
import re
from collections import defaultdict

from .labels import STRONG_STYLES
from .quality import text_flags
from .textnorm import normalize

# soundtags flags that make a clip unusable for training at any stage (subtitle says so: another speaker talks over
# it, the words are foreign or unintelligible, a word is bleeped, or the voice comes from a TV / PA / phone speaker)
EXCLUDE_FLAGS = {"overlap", "foreign_speech", "unintelligible", "censored", "channel"}
STAGE2_EXCLUDE_FLAGS = EXCLUDE_FLAGS | {"bg_laugh"}  # a laugh track under speech would teach [gülme] the wrong sound


def _sound_ok(r: dict, soundtags: dict[str, dict] | None, stage: int) -> bool:
    if soundtags is None:
        return True  # ttsdata soundtags has not run: nothing to exclude
    banned = STAGE2_EXCLUDE_FLAGS if stage >= 2 else EXCLUDE_FLAGS
    return not (set(soundtags.get(r["id"], {}).get("flags", [])) & banned)


def _gates(min_dur, max_dur, max_cer, max_music_db, max_music_db_separated, min_cluster, max_trail, soundtags, stage):
    return [
        ("audited", lambda r, q: q is not None and "text_flags" in q and "trail_sil" in q),  # fail closed
        ("duration", lambda r, q: min_dur <= r["duration"] <= max_dur),
        ("cer", lambda r, q: r.get("cer") is not None and r["cer"] <= max_cer),
        ("cluster", lambda r, q: (r.get("speaker_cluster_size") or 0) >= min_cluster),
        ("text", lambda r, q: not q.get("text_flags")),
        ("trail", lambda r, q: (q.get("trail_sil") or 0.0) <= max_trail),
        ("music", lambda r, q: q.get("music_db") is None or
         q["music_db"] <= (max_music_db_separated if q.get("source") == "separated" else max_music_db)),
        ("sound_flags", lambda r, q: _sound_ok(r, soundtags, stage)),
    ]


def tier_funnel(rows: list[dict], quality: dict[str, dict], *, min_dur: float = 3.0, max_dur: float = 20.0,
                max_cer: float = 0.05, max_music_db: float = -10.0, max_music_db_separated: float = 0.0,
                min_cluster: int = 10, max_trail: float = 0.5, soundtags: dict[str, dict] | None = None,
                stage: int = 1) -> list[tuple[str, list[dict]]]:
    """Rows surviving each gate, cumulatively: [("audited", rows), ("duration", rows), ...].

    Music gate (music_db from separate_vocals.py) matches separation.choose_source: originals up to -10 dB (the
    separator reports -12..-25 dB of residue on clean speech), separated vocals up to 0 dB; all five bands sounded
    clean in the 2026-10-28 listening check.
    soundtags: {clip id: soundtags.jsonl row} when `ttsdata soundtags` has run (None: no sound_flags gate);
    stage 2 (expressive training) also drops clips under a laugh track."""
    kept = rows
    out = []
    for name, ok in _gates(min_dur, max_dur, max_cer, max_music_db, max_music_db_separated, min_cluster, max_trail,
                           soundtags, stage):
        kept = [r for r in kept if ok(r, quality.get(r["id"]))]
        out.append((name, kept))
    return out


def select_tier(rows: list[dict], quality: dict[str, dict], **kw) -> list[dict]:
    """Rows that pass every gate (see tier_funnel for the gates and their defaults)."""
    funnel = tier_funnel(rows, quality, **kw)
    return funnel[-1][1] if funnel else []


def restored_only(rows: list[dict], restoration: dict[str, dict]) -> list[dict]:
    """Rows whose audio Sidon restored (restoration.jsonl source "sidon"). Fallbacks (music removed only, echo and
    separator artifacts left) and clips never restored are dropped."""
    return [r for r in rows if (restoration.get(r["id"]) or {}).get("source") == "sidon"]


def combine_units(singles: list[dict], merged: list[dict], quality: dict[str, dict], **tier_kw) -> list[dict]:
    """Training units: merged clips that pass the tier replace their members; every other single clip stays.

    Each returned row carries "root": "dataset" or "dataset_long", so the caller can resolve its audio path.
    (The tier is applied to the union again by the caller; merged rows pass it twice, harmlessly.)"""
    good_merged = select_tier(merged, quality, **tier_kw)
    covered = {m for row in good_merged for m in row["members"]}
    out = [{**r, "root": "dataset"} for r in singles if r["id"] not in covered]
    out += [{**r, "root": "dataset_long"} for r in good_merged]
    return out


_EVIDENCE_RANK = {"laugh_syllables": 3, "tagger": 2, "clean": 1, None: 0}


def combine_events(units: list[dict], events: list[dict], quality: dict[str, dict], evidence: set[str],
                   include_unconfirmed: bool = False, **tier_kw) -> list[dict]:
    """Event-extended clips (ttsdata extend-events / verify-events) replace their base clip in the units.

    An extended row is eligible when it is verified with an accepted evidence level (or unconfirmed and
    include_unconfirmed), passes the tier, and its base is among the units; one row per base (strongest evidence,
    then longest). Rows carry "root": "dataset_events"."""
    def eligible(e):
        d = e.get("detector") or {}
        if e.get("verified"):
            return d.get("evidence") in evidence
        return include_unconfirmed and d.get("reason") == "unconfirmed"

    good = select_tier([e for e in events if eligible(e)], quality, **tier_kw)
    unit_ids = {u["id"] for u in units}
    best: dict[str, dict] = {}
    for e in good:
        if e["base"] not in unit_ids:
            continue
        rank = (_EVIDENCE_RANK.get((e.get("detector") or {}).get("evidence"), 0), e["duration"])
        if e["base"] not in best or rank > best[e["base"]][0]:
            best[e["base"]] = (rank, e)
    out = [u for u in units if u["id"] not in best]
    out += [{**e, "root": "dataset_events"} for _, e in best.values()]
    return out


def restore_bases(rows: list[dict], events_ok: set[str], units: list[dict], quality: dict[str, dict], **tier_kw) -> list[dict]:
    """Stage 2: an event-extended row whose tag will not be written (rarer than --min-tag) would train an untagged
    laugh as plain speech, so it leaves the slice and its base clip (from the units before combine_events) returns in
    its place when that base passes the tier on its own."""
    by_id = {u["id"]: u for u in units}
    out = []
    for r in rows:
        if r.get("root") != "dataset_events" or r.get("tag") in events_ok:
            out.append(r)
            continue
        base = by_id.get(r.get("base"))
        if base is not None and select_tier([base], quality, **tier_kw):
            out.append(base)
    return out


def stage_filter(rows: list[dict], tags: dict[str, dict] | None, stage: int) -> list[dict]:
    """Stage 1 (plain prosody): no event-extended rows and no rows whose subtitle label marks a strong delivery
    (they would be trained with plain text that sounds emotional). Stage 2: everything."""
    if stage >= 2:
        return list(rows)
    out = []
    for r in rows:
        if r.get("root") == "dataset_events":
            continue
        style = set((tags or {}).get(r["id"], {}).get("style", []))
        if style & STRONG_STYLES:
            continue
        out.append(r)
    return out


def pair_references(rows: list[dict], ratio: float = 0.4, seed: int = 0) -> dict[str, str]:
    """For a seeded `ratio` of rows, a reference clip id from the same speaker cluster (never the row itself),
    preferring a clip with the same non-null speaker_hint, then clips of at least 3 s. Singleton clusters are skipped."""
    rng = random.Random(seed)
    by_spk: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_spk[r["speaker_id"]].append(r)
    eligible = [r for r in rows if len(by_spk[r["speaker_id"]]) >= 2]
    n = round(ratio * len(eligible))
    targets = sorted(rng.sample(eligible, n), key=lambda r: r["id"]) if n < len(eligible) else eligible
    pairs: dict[str, str] = {}
    for r in targets:
        pool = [c for c in by_spk[r["speaker_id"]] if c["id"] != r["id"]]
        if r.get("speaker_hint"):  # same named hint, else an unlabelled clip, never a clip labelled with another name
            same = [c for c in pool if c.get("speaker_hint") == r["speaker_hint"]]
            unlabelled = [c for c in pool if not c.get("speaker_hint")]
            pool = same or unlabelled or pool
        long_enough = [c for c in pool if c["duration"] >= 3.0]
        pool = long_enough or pool
        pairs[r["id"]] = rng.choice(sorted(pool, key=lambda c: c["id"]))["id"]
    return pairs


def replay_text(raw: str) -> str | None:
    """Replay-corpus transcript in the corpus' spoken form (digits expanded), or None when it would be flagged."""
    raw = raw.strip()
    if not raw:
        return None
    norm = normalize(raw)
    if not norm.strip() or text_flags(raw, norm):
        return None
    return norm


def manifest_row(row: dict, ref: dict | None, audio_root: str, dataset_id: int = 0,
                 ref_audio_root: str | None = None) -> dict:
    """One VoxCPM manifest row; ref_audio_root when the reference clip lives in another root (a merged clip)."""
    d = {"audio": f"{audio_root}/{row['id']}.wav", "text": row["normalized_text"], "duration": row["duration"],
         "dataset_id": dataset_id}
    if ref is not None:
        d["ref_audio"] = f"{ref_audio_root or audio_root}/{ref['id']}.wav"
        d["ref_duration"] = ref["duration"]
    return d


def validate_row(d: dict) -> list[str]:
    """Problems with a manifest row (empty list = valid for the VoxCPM trainer)."""
    errs = []
    if not isinstance(d.get("audio"), str) or not d["audio"]:
        errs.append("audio missing")
    if not isinstance(d.get("text"), str) or not d["text"].strip():
        errs.append("text empty")
    elif re.search(r"\d", d["text"]):
        errs.append("text has digits")
    dur = d.get("duration")
    if not isinstance(dur, (int, float)) or not (1.0 <= dur <= 30.0):
        errs.append("duration out of range (1–30 s)")
    has_ref, has_ref_dur = d.get("ref_audio") is not None, isinstance(d.get("ref_duration"), (int, float))
    if has_ref and not has_ref_dur:
        errs.append("ref_duration missing for ref_audio")
    if has_ref_dur and not has_ref:
        errs.append("ref_audio missing for ref_duration")
    if has_ref and d["ref_audio"] == d["audio"]:
        errs.append("ref_audio equals audio")
    if not isinstance(d.get("dataset_id"), int):
        errs.append("dataset_id must be int")
    return errs


def cap_hours(rows: list[dict], max_hours: float, max_series_share: float, seed: int,
              taken: list[dict] | None = None) -> list[dict]:
    """Seeded sample of rows totalling about max_hours, no series above max_series_share of it.

    taken: rows already chosen (stage 2 labels_first). They are always kept and count toward the total and toward
    their series' cap; the rest of the budget is filled at random from the other rows."""
    taken = list(taken or [])
    ids = {r["id"] for r in taken}
    rest = [r for r in rows if r["id"] not in ids] if taken else rows
    total = sum(r["duration"] for r in rest + taken) / 3600
    if max_hours <= 0 or total <= max_hours:
        return sorted(taken + rest, key=lambda r: r["id"]) if taken else rows
    rng = random.Random(seed)
    taken_h: dict[str, float] = defaultdict(float)
    for r in taken:
        taken_h[r["series"]] += r["duration"] / 3600
    by_series: dict[str, list[dict]] = defaultdict(list)
    for r in rest:
        by_series[r["series"]].append(r)
    hours = {s: sum(r["duration"] for r in v) / 3600 for s, v in by_series.items()}
    alloc = {s: min(h, max(0.0, max_series_share * max_hours - taken_h[s])) for s, h in hours.items()}
    budget = max(0.0, max_hours - sum(taken_h.values()))
    if sum(alloc.values()) > 0:
        scale = budget / sum(alloc.values())
        if scale < 1:
            alloc = {s: a * scale for s, a in alloc.items()}
    out = list(taken)
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


def labels_first(rows: list[dict], labels_of, max_hours: float, per_label: int, seed: int = 0) -> list[dict]:
    """Stage 2: rows that carry a label, taken before the random fill so rare labels are not sampled away.

    Rarest label first; per label at most per_label clips (a row already taken for another label counts toward this
    one too), seeded random choice within a label; stops once max_hours is spent, so the rare labels always get in."""
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        for lab in set(labels_of(r)):
            groups[lab].append(r)
    taken, ids, spent, budget = [], set(), 0.0, max_hours * 3600
    for lab in sorted(groups, key=lambda k: (len(groups[k]), k)):
        v = sorted(groups[lab], key=lambda r: r["id"])
        rng.shuffle(v)
        n = sum(1 for r in v if r["id"] in ids)
        for r in v:
            if n >= per_label:
                break
            if r["id"] in ids:
                continue
            if spent + r["duration"] > budget:
                return taken
            taken.append(r)
            ids.add(r["id"])
            spent += r["duration"]
            n += 1
    return taken
