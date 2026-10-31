"""Blind listening pages: which side is ours (A/B rounds), which style each clip had (style rounds), and the tallies
of the listener's answers. Pure functions; scripts/blind_page.py writes the HTML."""
from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path

UNANSWERED = ("-", "", None)


def ab_key(n: int, seed: int) -> list[str]:
    """Per pair, the side ("A" or "B") our system is on: half each, in seeded random order."""
    key = ["A"] * (n // 2) + ["B"] * (n - n // 2)
    random.Random(seed).shuffle(key)
    return key


def tally_ab(key: list[str], picks: list[str]) -> dict:
    """picks are "A", "B", "same" (or "=") per pair, "-" when unanswered; "unanswered" only appears when > 0."""
    out = {"ours": 0, "theirs": 0, "same": 0}
    missing = 0
    for side, pick in zip(key, picks):
        if pick in UNANSWERED:
            missing += 1
        elif pick in ("same", "="):
            out["same"] += 1
        else:
            out["ours" if pick == side else "theirs"] += 1
    if missing:
        out["unanswered"] = missing
    return out


def style_items(by_style: dict[str, list[str]], n_per: int, seed: int) -> list[dict]:
    """n_per clips from each style, shuffled so the same sentence (file name) never plays twice in a row."""
    rng = random.Random(seed)
    items = [{"wav": w, "style": s} for s, wavs in sorted(by_style.items()) for w in sorted(rng.sample(sorted(wavs), n_per))]
    sentence = lambda it: Path(it["wav"]).stem  # noqa: E731
    for _ in range(1000):
        rng.shuffle(items)
        if all(sentence(a) != sentence(b) for a, b in zip(items, items[1:])):
            return items
    raise ValueError("cannot order the clips without repeating a sentence back to back; use more sentences")


def tally_style(items: list[dict], picks: list[str]) -> dict:
    """Per style: correct picks and answered clips; confusion counts {(true style, picked): n}."""
    per: dict[str, dict] = {}
    conf: Counter = Counter()
    for it, pick in zip(items, picks):
        d = per.setdefault(it["style"], {"correct": 0, "n": 0})
        if pick in UNANSWERED:
            continue
        d["n"] += 1
        d["correct"] += pick == it["style"]
        conf[(it["style"], pick)] += 1
    return {"per_style": per, "confusion": dict(conf)}


def parse_answers(text: str) -> dict[int, str]:
    """'style check: 1:angry 2:- 3:sad' (the page's Copy answers line) -> {1: "angry", 2: "-", 3: "sad"}."""
    return {int(n): v for n, v in re.findall(r"(\d+):(\S+)", text)}
