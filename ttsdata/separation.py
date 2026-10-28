"""Decide, per clip, whether the vocal stem from a music separator should replace the original audio.

Separators (BS-Roformer and friends) also colour clean speech slightly, so a clip only switches to its
separated vocals when the instrumental stem is loud enough relative to the vocals to matter.
"""
from __future__ import annotations

import numpy as np

SILENT_DB = -80.0


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


def relative_loudness_db(vocals: np.ndarray, instrumental: np.ndarray) -> float:
    """Instrumental minus vocal level in dB (RMS). SILENT_DB when there is no instrumental; +80 when there is no voice."""
    ri, rv = _rms(instrumental), _rms(vocals)
    if ri < 1e-6:
        return SILENT_DB
    if rv < 1e-6:
        return -SILENT_DB
    return float(20.0 * np.log10(ri / rv))


def choose_source(music_db: float, threshold_db: float = -10.0, vocal_ratio: float | None = None,
                  min_vocal_ratio: float = 0.25) -> str:
    """'separated' when the instrumental sits at or above threshold_db relative to the vocals, else 'original'.

    The separator leaves a -12..-25 dB "instrumental" residue even on clean read speech, hence the -10 dB default.
    The original is kept when the stems are unreliable: instrumental louder than the voice (music_db > 0, the voice
    went into the wrong stem) or the vocal stem holding under min_vocal_ratio of the original's RMS."""
    if music_db > 0.0:
        return "original"
    if vocal_ratio is not None and vocal_ratio < min_vocal_ratio:
        return "original"
    return "separated" if music_db >= threshold_db else "original"


def loudness_normalize(y: np.ndarray, sr: int, target_lufs: float = -20.0, peak_ceiling: float = 0.99) -> np.ndarray:
    """Same rule as the build stage: integrated loudness to target, then scale down if the peak exceeds the ceiling."""
    import pyloudnorm as pyln

    meter = pyln.Meter(sr)
    try:
        loud = meter.integrated_loudness(y.astype(np.float64))
    except ValueError:
        loud = float("nan")
    if np.isfinite(loud):
        y = pyln.normalize.loudness(y, loud, target_lufs)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > peak_ceiling:
        y = y * (peak_ceiling / peak)
    return y.astype(np.float32)


def manifest_ids(rows: list[dict], root_name: str) -> set[str]:
    """Clip ids a VoxCPM manifest uses from one dataset root (its `audio` and `ref_audio` paths, any wav folder):
    lets separate_vocals.py clean only the clips a training set needs instead of the whole build."""
    ids = set()
    for d in rows:
        for key in ("audio", "ref_audio"):
            parts = (d.get(key) or "").split("/")
            if len(parts) >= 3 and parts[-3] == root_name and parts[-1].endswith(".wav"):
                ids.add(parts[-1][:-4])
    return ids
