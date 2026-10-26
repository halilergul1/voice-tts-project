"""Parse WebVTT / SRT subtitle files into cleaned speech segments.

The "Engelsiz" (accessible) broadcasts carry SDH-style subtitles:
  - sound descriptions in parentheses:  (Jenerik müziği), (Siren çalıyor)
  - off-screen speaker labels:          (Usta) Gaz ver.
  - sentence continuation across cues:  "Belki ölümden..." / "...korkmadığı için."
  - two speakers in one cue, dash-led:  "-Gel.\n-Geliyorum."
This module strips the non-speech material and keeps one speaker's speech per segment.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

TS_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{3})")
TS_SHORT_RE = re.compile(r"^(\d{1,2}):(\d{2})[.,](\d{3})$")
TAG_RE = re.compile(r"<[^>]+>")
PAREN_FULL_RE = re.compile(r"^[\(\[][^\)\]]*[\)\]][.!?,]*$")
PAREN_LEAD_RE = re.compile(r"^[\(\[]([^\)\]]{1,40})[\)\]][:\s]*")
PAREN_INLINE_RE = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]\s*")
UPPER_LABEL_RE = re.compile(r"^([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ .']{1,30}):\s+")
DASH_LEAD_RE = re.compile(r"^[-–—]\s*")
ELL_END_RE = re.compile(r"(\.\.\.|…)[\"']?\s*$")
ELL_START_RE = re.compile(r"^\s*[\"']?(\.\.\.|…)\s*")
TERMINAL_RE = re.compile(r"[.!?…\"']\s*$")
MUSIC_CHARS = "♪♫♬♩"
PAREN_BLOCK_RE = re.compile(r"\([^()]*\)|\[[^\[\]]*\]", re.DOTALL)
MUSIC_WORDS_RE = re.compile(r"müzi|şarkı|çalıyor|çalmaya|söylüyor|söylen|melodi|jenerik|ezgi", re.IGNORECASE)
CHORUS_RE = re.compile(r"hep bir|birlikte|herkes|hepsi|koro|beraber", re.IGNORECASE)
QUOTED_RE = re.compile(r"^[\"“„»].*[\"”“«]\s*(\.\.\.)?$", re.DOTALL)


@dataclass
class Cue:
    start: float
    end: float
    lines: list[str]


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker_hint: str | None = None
    flags: list[str] = field(default_factory=list)
    n_cues: int = 1


def _ts(s: str) -> float:
    s = s.strip()
    m = TS_RE.match(s)
    if m:
        h, mi, se, ms = map(int, m.groups())
        return h * 3600 + mi * 60 + se + ms / 1000
    m = TS_SHORT_RE.match(s)
    if m:
        mi, se, ms = map(int, m.groups())
        return mi * 60 + se + ms / 1000
    raise ValueError(f"bad timestamp {s!r}")


def parse(path: str | Path) -> list[Cue]:
    """Parse a .vtt or .srt file, line by line.

    Tolerates BOMs anywhere, CRLF, VTT positioning tags, VTT cue ids, SRT indices and
    cues glued together without the separating blank line.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace").replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cues: list[Cue] = []
    cur: Cue | None = None

    def close():
        nonlocal cur
        if cur is not None:
            cur.lines = [l for l in cur.lines if l.strip()]
            if cur.lines and cur.end > cur.start:
                cues.append(cur)
        cur = None

    for raw in text.split("\n"):
        line = raw.strip()
        if "-->" in line:
            if cur is not None and cur.lines and cur.lines[-1].strip().isdigit():
                cur.lines.pop()  # SRT index glued onto the previous cue
            close()
            left, right = line.split("-->", 1)
            try:
                cur = Cue(_ts(left), _ts(right.strip().split()[0]), [])
            except (ValueError, IndexError):
                cur = None
        elif not line:
            close()
        elif cur is not None:
            cur.lines.append(line)
    close()

    out: list[Cue] = []
    for c in cues:
        if out and out[-1].start == c.start and out[-1].end == c.end and out[-1].lines == c.lines:
            continue
        out.append(c)
    return out


def clean_cue(cue: Cue) -> Segment | None:
    """Return a Segment or None when the cue carries no usable speech."""
    lines = [html.unescape(TAG_RE.sub("", l)).strip() for l in cue.lines]
    lines = [l for l in lines if l]
    if not lines:
        return None
    if any(ch in l for l in lines for ch in MUSIC_CHARS):
        return None  # sung lyrics

    # speaker labels: a short parenthetical / UPPER: label at the start of a line, followed by text on that line
    labels: list[str] = []
    stripped: list[str] = []
    for l in lines:
        m = PAREN_LEAD_RE.match(l) or UPPER_LABEL_RE.match(l)
        if m and l[m.end():].strip() and not MUSIC_WORDS_RE.search(m.group(1)):
            labels.append(m.group(1).strip())
            l = l[m.end():].strip()
        stripped.append(l)

    joined = "\n".join(stripped)
    descriptions = PAREN_BLOCK_RE.findall(joined)
    music_context = any(MUSIC_WORDS_RE.search(d) for d in descriptions)
    joined = PAREN_BLOCK_RE.sub(" ", joined)
    kept = [re.sub(r"\s+", " ", l).strip() for l in joined.split("\n")]
    kept = [l for l in kept if l]
    if not kept:
        return None  # pure sound description
    text = " ".join(kept).strip()
    if QUOTED_RE.match(text) or (music_context and text.startswith(('"', "“", "„"))):
        return None  # song lyrics rendered in quotes

    flags: list[str] = []
    if len(set(labels)) >= 2 or any(CHORUS_RE.search(lb) for lb in labels):
        flags.append("multi_speaker")
    dash_lines = sum(1 for l in kept if DASH_LEAD_RE.match(l))
    if dash_lines >= 2 or (dash_lines == 1 and len(kept) > 1 and not DASH_LEAD_RE.match(kept[0])):
        flags.append("multi_speaker")
    kept = [DASH_LEAD_RE.sub("", l) for l in kept]
    text = re.sub(r"\s+", " ", " ".join(kept)).strip()
    if not text:
        return None
    speaker = labels[0] if len(set(labels)) == 1 else None
    return Segment(cue.start, cue.end, text, speaker, sorted(set(flags)))


def merge_continuations(segs: list[Segment], max_gap: float, max_gap_comma: float, max_dur: float) -> list[Segment]:
    """Join cues that split one sentence ("..." / trailing comma) when the gap is small."""
    out: list[Segment] = []
    for s in segs:
        if out:
            p = out[-1]
            gap = s.start - p.end
            same_speaker = p.speaker_hint is None or s.speaker_hint is None or p.speaker_hint == s.speaker_hint
            ok_flags = "multi_speaker" not in p.flags and "multi_speaker" not in s.flags
            ellipsis_join = bool(ELL_END_RE.search(p.text)) and bool(ELL_START_RE.match(s.text)) and gap <= max_gap
            open_join = (not TERMINAL_RE.search(p.text)) and (not ELL_START_RE.match(s.text)) and gap <= max_gap_comma
            if same_speaker and ok_flags and (s.end - p.start) <= max_dur and (ellipsis_join or open_join):
                left = ELL_END_RE.sub("", p.text).rstrip() if ellipsis_join else p.text.rstrip()
                right = ELL_START_RE.sub("", s.text).lstrip() if ellipsis_join else s.text.lstrip()
                p.text = f"{left} {right}"
                p.end = s.end
                p.n_cues += 1
                p.speaker_hint = p.speaker_hint or s.speaker_hint
                if "merged" not in p.flags:
                    p.flags.append("merged")
                continue
        out.append(replace(s, flags=list(s.flags)))
    for s in out:
        if ELL_START_RE.match(s.text):
            s.text = ELL_START_RE.sub("", s.text)
            s.flags.append("fragment_start")
        if ELL_END_RE.search(s.text):
            s.flags.append("fragment_end")
    return [s for s in out if s.text.strip()]


def load_segments(path: str | Path, max_gap: float, max_gap_comma: float, max_dur: float) -> tuple[list[Segment], dict]:
    cues = parse(path)
    segs: list[Segment] = []
    dropped = {"non_speech_cue": 0}
    for c in cues:
        s = clean_cue(c)
        if s is None:
            dropped["non_speech_cue"] += 1
        else:
            segs.append(s)
    merged = merge_continuations(segs, max_gap, max_gap_comma, max_dur)
    stats = {"cues": len(cues), "speech_cues": len(segs), "segments": len(merged), **dropped}
    return merged, stats
