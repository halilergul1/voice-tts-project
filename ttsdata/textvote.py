"""Task 3D text vote: which text a clip is trained on when Whisper and the subtitle disagree, decided by a second
recogniser of another model family."""
from __future__ import annotations

import jiwer

from ttsdata.textnorm import for_cer


def _cer(ref: str, hyp: str) -> float:
    ref, hyp = for_cer(ref or ""), for_cer(hyp or "")
    return jiwer.cer(ref, hyp) if ref and hyp else 1.0


def vote(subtitle: str, whisper: str, asr2: str, agree: float = 0.05) -> str:
    """"subtitle" when the second recogniser matches the subtitle (Whisper misheard), "spoken" when it matches Whisper
    instead (the subtitle cleaned up what was said: train on the recognisers' text), "drop" when it matches neither.
    Texts are compared after for_cer (lowercase, no punctuation, digits as words); agree is the largest CER."""
    if _cer(subtitle, asr2) <= agree:
        return "subtitle"
    if _cer(whisper, asr2) <= agree:
        return "spoken"
    return "drop"
