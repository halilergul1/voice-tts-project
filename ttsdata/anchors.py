"""Stage 2 anchor set: plain sentences spoken by the base model itself (stage 1 inference setting, clip B), mixed into
training so a LoRA that learns styles keeps the base model's plain voice. Pure helpers; scripts/make_anchor_set.py
does the I/O."""
from __future__ import annotations

import random


def pick_sentences(rows: list[dict], exclude: set[str], n: int, seed: int,
                   min_dur: float = 3.0, max_dur: float = 20.0) -> list[str]:
    """`normalized_text` of train-split, unlabelled tier rows of min_dur-max_dur s with no `text_flags`, not in
    `exclude` (the eval sentences), deduplicated, then a seeded sample of at most n."""
    seen, pool = set(exclude), []
    for r in rows:
        t = (r.get("normalized_text") or "").strip()
        if (not t or t in seen or r.get("split") != "train" or r.get("labelled") or r.get("text_flags")
                or not min_dur <= r["duration"] <= max_dur):
            continue
        seen.add(t)
        pool.append(t)
    pool.sort()
    random.Random(seed).shuffle(pool)
    return pool[:n]


def anchor_keep(score: dict, min_ref_cos: float = 0.75) -> bool:
    """A synthesised anchor is kept when every word came out (wer 0), no prompt word was read aloud, the voice stays
    near the run centre (spk_cos; high-voice drift shows up here) and UTMOS scored it."""
    cos = score.get("spk_cos")
    return (score.get("wer") == 0 and not score.get("read_aloud") and cos is not None and cos >= min_ref_cos
            and score.get("utmos") is not None)
