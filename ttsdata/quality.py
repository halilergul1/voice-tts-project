"""Per-clip quality signals for the built dataset (used by scripts/audit_quality.py and the manifest tier).

Pure functions only: text problems a TTS model would read wrong, leading / trailing silence, per-episode
CER outliers. Audio-model scores (UTMOS) and music ratios are added by the scripts that own those models.
"""
from __future__ import annotations

import re

import numpy as np

_TR_UPPER = "A-ZÇĞİÖŞÜ"
_TR_LOWER = "a-zçğıöşü"
ABBREVIATIONS = {"vs.", "vb.", "dr.", "prof.", "av.", "bkz.", "no.", "tel.", "doç.", "yrd.", "sok.", "cad.",
                 "apt.", "mah.", "alb.", "yzb.", "bşk.", "müh.", "op.", "hz.", "st."}
ENGLISH_STOPWORDS = {"the", "and", "you", "with", "what", "this", "that", "for", "your", "are", "have"}  # "not" is Turkish too
NUMBER_WORDS = {"sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz", "on", "yirmi", "otuz", "kırk",
                "elli", "altmış", "yetmiş", "seksen", "doksan", "yüz", "bin", "milyon", "milyar", "buçuk"}
_LEFTOVER = re.compile(r"[()\[\]♪*#<>]")
_ALLCAPS = re.compile(rf"(?<![{_TR_UPPER}{_TR_LOWER}])[{_TR_UPPER}]{{2,}}(?![{_TR_LOWER}])")
_FOREIGN_LETTER = re.compile(r"[wxqWXQ]")
_WORD = re.compile(rf"[{_TR_UPPER}{_TR_LOWER}]+")


def text_flags(text: str, normalized_text: str) -> list[str]:
    """Flags for text a TTS model is likely to read wrong: digits, abbrev, allcaps, leftover, fragment, latin."""
    flags = set()
    t = normalized_text.strip()
    if re.search(r"\d", t):
        flags.add("digits")
    tokens = t.split()
    for i, tok in enumerate(tokens):
        low = tok.lower()
        if low in ABBREVIATIONS:
            flags.add("abbrev")
        elif tok.endswith(".") and i < len(tokens) - 1:
            core = tok[:-1]
            if re.fullmatch(rf"[{_TR_UPPER}]", core):
                flags.add("abbrev")  # an initial: "A. Kadir", "M. Kemal"
            elif re.fullmatch(rf"[{_TR_UPPER}{_TR_LOWER}]{{1,4}}", core) and tokens[i + 1][:1].islower() \
                    and tokens[i + 1].strip(".,!?").lower() not in NUMBER_WORDS:
                flags.add("abbrev")  # "ünv. kampüsü": a period followed by a lowercase word is not a sentence end
                # (num2words writes expanded numbers in lowercase after a sentence end: "Üç yıl. on sekiz yaşına")
    if _ALLCAPS.search(t):
        flags.add("allcaps")
    if _LEFTOVER.search(t) or _LEFTOVER.search(text):
        flags.add("leftover")
    if t.startswith(("...", "…")) or t.endswith(("...", "…")):
        flags.add("fragment")
    for w in _WORD.findall(t):
        if _FOREIGN_LETTER.search(w) or w.lower() in ENGLISH_STOPWORDS:
            flags.add("latin")
            break
    return sorted(flags)


def silence_edges(y: np.ndarray, sr: int, thresh_db: float = -40.0, hop: float = 0.02) -> tuple[float, float]:
    """Seconds of near-silence at the start and at the end of a clip (frame RMS below thresh_db re full scale)."""
    n = int(sr * hop)
    if n <= 0 or len(y) == 0:
        return 0.0, 0.0
    frames = len(y) // n
    if frames == 0:
        return len(y) / sr, len(y) / sr
    x = y[: frames * n].astype(np.float64).reshape(frames, n)
    db = 20.0 * np.log10(np.sqrt(np.mean(x * x, axis=1)) + 1e-9)
    above = np.flatnonzero(db > thresh_db)
    if len(above) == 0:
        return len(y) / sr, len(y) / sr
    lead = above[0] * hop
    trail = len(y) / sr - (above[-1] + 1) * hop
    return float(lead), float(max(trail, 0.0))


def episode_outliers(rows: list[dict], min_mad: float = 0.005) -> list[dict]:
    """Episodes whose mean CER sits above median + 2 * MAD of the per-episode means (worst first)."""
    per: dict[str, list[float]] = {}
    for r in rows:
        if r.get("cer") is not None:
            per.setdefault(r["uid"], []).append(float(r["cer"]))
    if not per:
        return []
    means = {uid: float(np.mean(v)) for uid, v in per.items()}
    vals = np.array(list(means.values()))
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    thr = med + 2.0 * max(mad, min_mad)
    out = [{"uid": uid, "mean_cer": round(m, 4), "n": len(per[uid])} for uid, m in means.items() if m > thr]
    return sorted(out, key=lambda o: -o["mean_cer"])
