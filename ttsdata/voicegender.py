"""Speaker gender from voice pitch, for voice descriptions in training text ("(bir kadın sesi) ...").

Per clip: median F0 of the voiced frames. Per speaker id: the median over its clips, and how many clips agree.
Adult male speech sits around 85-155 Hz, adult female around 165-255 Hz; children and raised voices blur this, so
a speaker only gets a label when its median is clearly on one side and most of its clips agree. Unlabelled speakers
simply get no description.
"""
from __future__ import annotations

import numpy as np

MALE_MAX_HZ = 150.0
FEMALE_MIN_HZ = 175.0
CLIP_SPLIT_HZ = 165.0
DESCRIPTIONS = {
    "tr": {"male": "(bir erkek sesi)", "female": "(bir kadın sesi)"},
    "en": {"male": "(A man's voice)", "female": "(A woman's voice)"},
}
DESCRIPTION = DESCRIPTIONS["tr"]


def clip_f0(y: np.ndarray, sr: int) -> float | None:
    """Median F0 (Hz) over voiced frames, None when nothing is voiced."""
    import librosa

    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr, frame_length=1024)
    f0 = f0[voiced] if voiced is not None else np.array([])
    f0 = f0[~np.isnan(f0)]
    return float(np.median(f0)) if len(f0) else None


def speaker_gender(f0s: list[float | None], min_clips: int = 3, min_agree: float = 0.75) -> dict:
    """{"gender": "male" | "female" | None, "f0": median Hz, "n": clips measured, "agree": share on the majority side}."""
    vals = np.array([f for f in f0s if f is not None], dtype=float)
    if len(vals) < min_clips:
        return {"gender": None, "f0": float(np.median(vals)) if len(vals) else None, "n": int(len(vals)), "agree": None}
    med = float(np.median(vals))
    high = float(np.mean(vals > CLIP_SPLIT_HZ))
    agree = max(high, 1.0 - high)
    gender = None
    if agree >= min_agree:
        if med <= MALE_MAX_HZ and high < 0.5:
            gender = "male"
        elif med >= FEMALE_MIN_HZ and high > 0.5:
            gender = "female"
    return {"gender": gender, "f0": round(med, 1), "n": int(len(vals)), "agree": round(agree, 3)}


def describe(text: str, gender: str | None, descriptions: dict[str, str] = DESCRIPTION) -> str:
    """Voice-design prefix for a training row (VoxCPM2 format: "(description) text"). Text that already starts with
    a parenthesised style ("(neşeyle) ...", stage 2) is left alone; so is a speaker with no gender label."""
    if gender not in descriptions or text.startswith("("):
        return text
    return f"{descriptions[gender]} {text}"
