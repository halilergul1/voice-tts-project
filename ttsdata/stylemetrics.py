"""Scoring helpers for stage 2 (styles and sound tags): label-aware word error inputs, label words read aloud,
emotion2vec+ style hits and laugh realisation. Pure functions; the models run in scripts/score_styles.py."""
from __future__ import annotations

import re

from ttsdata.emotion import COARSE
from ttsdata.eventverify import laugh_syllables
from ttsdata.labels import EVENT_TAG, STYLE_PREFIX
from ttsdata.textnorm import for_cer

_PREFIX_RE = re.compile(r"^\s*\([^)]*\)\s*")
_TAG_RE = re.compile(r"\[[^\]]*\]")
LABEL_WORDS = tuple(dict.fromkeys(for_cer(v[1:-1]) for v in (*STYLE_PREFIX.values(), *EVENT_TAG.values())))


def strip_labels(text: str) -> str:
    """'(öfkeyle) Neden geldin?' -> 'Neden geldin?'; '[gülme] Hadi. [nefes]' -> 'Hadi.' (only a leading prefix)."""
    t = _TAG_RE.sub(" ", _PREFIX_RE.sub("", text, count=1))
    return re.sub(r"\s+", " ", t).strip()


def scoring_texts(reference: str, hypothesis: str) -> tuple[str, str]:
    """(reference, hypothesis) for WER/CER: labels stripped from the reference; when the reference carried a tag,
    laughter tokens ("hahaha") are dropped from the hypothesis, since the sound was asked for."""
    ref = for_cer(strip_labels(reference))
    hyp = for_cer(hypothesis or "")
    if _TAG_RE.search(reference):
        hyp = " ".join(t for t in hyp.split() if laugh_syllables(t) < 2)
    return ref, hyp


def read_aloud(hypothesis: str, reference: str, extra_words: tuple[str, ...] = ()) -> list[str]:
    """Label words the model spoke as words: present in the hypothesis but not in the label-free reference."""
    hyp = f" {for_cer(hypothesis or '')} "
    ref = f" {for_cer(strip_labels(reference))} "
    found = []
    for w in dict.fromkeys((*LABEL_WORDS, *(for_cer(x) for x in extra_words))):
        if w and f" {w} " in hyp and f" {w} " not in ref:
            found.append((hyp.index(f" {w} "), w))
    return [w for _, w in sorted(found)]


def _coarse_top(post: dict[str, float]) -> str:
    return max(COARSE, key=lambda k: post.get(k, 0.0))


def style_hit(post: dict[str, float], target: str) -> bool:
    """emotion2vec+ posterior's top class among angry / happy / sad / neutral equals the target."""
    return _coarse_top(post) == target


def style_summary(rows: list[dict], target: str) -> dict:
    """n, hit rate, mean target posterior and share heard as neutral over rows carrying "e2v" posteriors."""
    posts = [r["e2v"] for r in rows if r.get("e2v")]
    n = len(posts)
    if not n:
        return {"n": 0, "hit_rate": None, "target_post": None, "neutral_rate": None}
    return {"n": n,
            "hit_rate": sum(style_hit(p, target) for p in posts) / n,
            "target_post": sum(p.get(target, 0.0) for p in posts) / n,
            "neutral_rate": sum(_coarse_top(p) == "neutral" for p in posts) / n}


def tag_realised(asr: str, tag: str) -> bool | None:
    """Laugh: Whisper wrote at least two ha-like syllables. Other tags have no detector (None: judge by ear)."""
    if tag == "laugh":
        return laugh_syllables(asr or "") >= 2
    return None
