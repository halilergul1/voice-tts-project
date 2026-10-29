"""Speech restoration (Sidon) for training clips: helpers for cutting clips with episode context and checking endings.

Sidon (sarulab-speech/sidon-v0.1) removes room echo and the robotic separator sound, but it fades a last word that the
subtitle cut stops on. Cutting the clip from the episode with CTX seconds on both sides (music removed), restoring
that, and trimming back keeps those endings; clips whose last word still comes out quieter than in the music-removed
clip fall back to it. Trial on 46 clips (docs/superpowers/plans/stage1_results.md): WER 7.7% vs 7.9% without Sidon,
PESQ estimate 2.29 -> 3.56.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

AUDIO_EXTS = {".m4a", ".webm", ".mp3", ".mp4", ".wav", ".opus", ".ogg", ".flac", ".aac", ".mkv"}
MAX_END_DROP_DB = 6.0


def episode_audio(ep_dir: Path) -> Path | None:
    """The downloaded episode audio (audio.<ext>); subtitle files such as audio.tr.vtt are not audio."""
    return next((p for p in sorted(Path(ep_dir).glob("audio.*")) if p.suffix.lower() in AUDIO_EXTS), None)


def context_span(start: float, end: float, ctx: float, total: float) -> tuple[float, float, float]:
    """(t0, t1, lead): the clip widened by ctx seconds on each side, clamped to the episode; lead = start - t0."""
    t0 = max(0.0, start - ctx)
    t1 = min(total, end + ctx)
    return round(t0, 3), round(t1, 3), round(start - t0, 3)


def trim_span(y: np.ndarray, sr: int, lead: float, dur: float) -> np.ndarray:
    """The clip's own span out of a context cut."""
    a = int(round(lead * sr))
    return y[a: a + int(round(dur * sr))]


def noise_pad(x: np.ndarray, n: int, rng: np.random.Generator, level: float = 1e-3) -> np.ndarray:
    """n samples of faint noise on both sides (Sidon keeps endings better than with digital silence)."""
    pad = lambda: (rng.standard_normal(n) * level).astype(x.dtype)  # noqa: E731
    return np.concatenate([pad(), x, pad()])


def rms_match(y: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return y * (np.sqrt(np.mean(np.square(ref, dtype=np.float64))) / (np.sqrt(np.mean(np.square(y, dtype=np.float64))) + 1e-12))


def _frames_db(y: np.ndarray, sr: int, hop_s: float = 0.01) -> np.ndarray:
    n = max(1, int(sr * hop_s))
    m = len(y) // n
    if m == 0:
        return np.array([-120.0])
    return 20 * np.log10(np.sqrt(np.mean(y[: m * n].reshape(m, n).astype(np.float64) ** 2, axis=1)) + 1e-6)


def last_word_db(y: np.ndarray, sr: int, window_s: float = 0.4, floor_db: float = 30.0) -> float:
    """Level of the last window_s of speech, relative to the clip's speech level (90th percentile of 10 ms frames).
    Trailing frames more than floor_db under that level count as silence and are skipped."""
    e = _frames_db(y, sr)
    ref = np.percentile(e, 90)
    voiced = np.where(e > ref - floor_db)[0]
    end = voiced[-1] + 1 if len(voiced) else len(e)
    w = max(1, int(window_s / 0.01))
    return float(np.mean(e[max(0, end - w): end]) - ref)


def ending_drop_db(restored: np.ndarray, sr_restored: int, reference: np.ndarray, sr_reference: int) -> float:
    """How much quieter the restored clip's last word is than the reference's (negative = quieter)."""
    return last_word_db(restored, sr_restored) - last_word_db(reference, sr_reference)


FIRST_GATE_CER = 0.08
MAX_EXTRA_CER = 0.05


def word_check(cer_sidon: float, cer_build: float | None, cer_clean, first_gate: float = FIRST_GATE_CER,
               max_extra: float = MAX_EXTRA_CER) -> tuple[str, float | None]:
    """Did Sidon change words? ('sidon' | 'fallback_words', CER of the music-removed clip or None).
    cer_sidon: Whisper on the restored clip vs the subtitle. When it is within first_gate of the build's CER the clip
    passes without a second transcription; otherwise cer_clean() (Whisper on the music-removed clip) decides: fall back
    when Sidon is more than max_extra worse. Clips that already missed words before Sidon are not punished for it."""
    if cer_sidon - (cer_build or 0.0) <= first_gate:
        return "sidon", None
    c = cer_clean()
    return ("fallback_words" if cer_sidon - c > max_extra else "sidon"), c
