"""Emotion labels from the Engelsiz audio description (stage 2 Task 3B).

The TRT describer speaks between dialogue lines and says what characters feel ("Şivenaz zindanda sinirle dört
dönerken"). This module finds those gaps, turns the describer's words into short spans, parses emotion words with the
character name in front of them, maps names to speaker clusters through the subtitles' speaker hints, and links a
description to that character's next lines. Pure functions; the Whisper pass is scripts/transcribe_narration.py, the
labelling pass scripts/narration_labels.py.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from ttsdata.eventverify import HALLUCINATION_RE
from ttsdata.soundtags import tr_lower

EMOTION_STEMS: dict[str, tuple[str, ...]] = {
    "angry": ("sinir", "öfke", "kızgın", "hiddet", "bağır", "terslen"),
    "sad": ("ağla", "gözyaş", "hüzün", "üzgün", "üzül", "hıçkır", "buruk"),
    "happy": ("gülüms", "gülerek", "gülüyor", "sevin", "neşe", "mutlu", "keyif", "kahkaha"),
    "scared": ("kork", "dehşet", "panik", "ürk"),
    "surprised": ("şaşkın", "şaşır", "hayretle"),
}
# crying while talking is its own class (checked first), so it can become a style of its own if it has the volume
CRYING_STEMS = ("ağlıyor", "ağlayarak", "hıçkır")
_CLASSES = (("crying", CRYING_STEMS), *EMOTION_STEMS.items())
# words that start with an emotion stem but mean something else ("korkuluk" = railing, "bağırsak" = intestine)
NOT_EMOTION = ("korkuluk", "bağırsak")

# after a verb stem: negative -ma/-me + tense or converb ("ağlamadı", "gülümsemeden", "ağlamamak", "korkmayan"),
# negative progressive ("ağlamıyor"), or the privative -sız ("keyifsiz", "korkusuz"); an optional stem vowel first,
# since "gülüms" stands for "gülümse-" / "gülümsü-" ("gülümsemeden"). The positive infinitive and verbal noun
# ("ağlamak", "ağlamaya") also start with -ma/-me and stay positive.
_NEG_RE = re.compile(r"^(?:[ae]?(?:m[ae](?:z|d[ıiuü]|dan|den|mak|mek|y[ae]n|y[ae]rak|m[ıiuü]ş|s[ıi]z)|m[ıiuü]yor)|[ıiuü]?s[ıiuü]z)")
_SENTENCE_RE = re.compile(r"[.!?;…]+")
_TOKEN_RE = re.compile(r"[\w'’]+", re.UNICODE)
_APOSTROPHE_RE = re.compile(r"['’]")
# the describer's intro / outro ("Bu dizinin betimlemesi TRT tarafından Sesli Betimleme Derneğine yaptırılmıştır")
_BOILERPLATE_RE = re.compile(r"betimle|izlediğiniz için")


def _emotion(token: str) -> str | None:
    low = tr_lower(token)
    if low.startswith(NOT_EMOTION):
        return None
    for emotion, stems in _CLASSES:
        for stem in stems:
            if low.startswith(stem):
                return None if _NEG_RE.match(low[len(stem):]) else emotion
    return None


def _name_before(tokens: list[str], i: int) -> str | None:
    """The run of capitalised words nearest before tokens[i], case suffixes after an apostrophe removed."""
    j = i - 1
    while j >= 0 and not tokens[j][0].isupper():
        j -= 1
    if j < 0:
        return None
    k = j
    while k - 1 >= 0 and tokens[k - 1][0].isupper():
        k -= 1
    return " ".join(_APOSTROPHE_RE.split(t)[0] for t in tokens[k:j + 1])


def parse_description(text: str) -> list[dict]:
    """[{name, emotion, word}] per emotion word, one row per emotion and sentence (the first word wins); name = the
    capitalised word(s) before it in the same sentence ("Ayşe'nin" -> "Ayşe"), None when there are none. A capitalised
    word inside a sentence is a name ("Sevinç", "Neşe"), never an emotion word."""
    out = []
    for sentence in _SENTENCE_RE.split(text):
        tokens = _TOKEN_RE.findall(sentence)
        seen = set()
        for i, tok in enumerate(tokens):
            emotion = None if i > 0 and tok[0].isupper() else _emotion(tok)
            if emotion and emotion not in seen:
                seen.add(emotion)
                out.append({"name": _name_before(tokens, i), "emotion": emotion, "word": tok})
    return out


def name_clusters(rows: list[dict], min_lines: int = 5, min_share: float = 0.6) -> dict[tuple[str, str], str]:
    """(series, name) -> speaker_id when that name's speaker_hint lines fall mostly in one speaker cluster."""
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for r in rows:
        hint, spk = (r.get("speaker_hint") or "").strip(), r.get("speaker_id")
        if hint and spk:
            counts[(r.get("series"), hint)][spk] += 1
    out = {}
    for key, c in counts.items():
        total = sum(c.values())
        spk, n = c.most_common(1)[0]
        if total >= min_lines and n / total >= min_share:
            out[key] = spk
    return out


def link_lines(desc: dict, lines: list[dict], cluster: str | None, before: float = 2.0, after: float = 20.0) -> list[str]:
    """Ids of the lines a description labels. Named and resolved: lines of `cluster` starting within
    [end - before, end + after] and before desc["next"] (the same character's next description). Otherwise only the
    first line of any speaker starting within 5 s after the description."""
    end = desc["end"]
    lines = sorted(lines, key=lambda l: l["start"])
    if desc.get("name") and cluster:
        stop = desc.get("next", float("inf"))
        return [l["id"] for l in lines
                if l.get("speaker_id") == cluster and end - before <= l["start"] <= end + after and l["start"] < stop]
    first = next((l for l in lines if end <= l["start"] <= end + 5.0), None)
    return [first["id"]] if first else []


def narration_gaps(cues: list[tuple[float, float]], total: float, min_gap: float = 1.5,
                   pad: float = 0.2) -> list[tuple[float, float]]:
    """Stretches of the episode with no speech cue: overlapping cues merge, each gap loses `pad` seconds next to a cue
    (dialogue spills past subtitle times), and gaps shorter than `min_gap` after that are dropped."""
    merged: list[list[float]] = []
    for a, b in sorted(cues):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    edges = [0.0] + [x for m in merged for x in m] + [total]
    out = []
    for a, b in zip(edges[::2], edges[1::2]):
        a2 = a + pad if a > 0 else a
        b2 = b - pad if b < total else b
        if b2 - a2 >= min_gap:
            out.append((round(a2, 3), round(b2, 3)))
    return out


def spans_from_words(words: list[dict], max_pause: float = 1.5) -> list[dict]:
    """Group Whisper words ({word, start, end}) into spans at pauses longer than `max_pause`; known Whisper
    hallucinations are removed, spans left without words and the describer's intro / outro ("... betimlemesi ...",
    "izlediğiniz için teşekkür ederiz") dropped. Returns [{start, end, text}]."""
    groups: list[list[dict]] = []
    for w in words:
        if groups and w["start"] - groups[-1][-1]["end"] <= max_pause:
            groups[-1].append(w)
        else:
            groups.append([w])
    out = []
    for g in groups:
        text = re.sub(r"\s+", " ", HALLUCINATION_RE.sub(" ", " ".join(w["word"].strip() for w in g))).strip()
        if re.search(r"\w\w", text) and not _BOILERPLATE_RE.search(tr_lower(text)):
            out.append({"start": round(g[0]["start"], 3), "end": round(g[-1]["end"], 3), "text": text})
    return out
