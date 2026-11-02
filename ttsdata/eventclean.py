"""Checks that an event clip's sound (cry, laugh, breath...) comes from the line's own speaker and is not covered by
the audio-description narrator. A flagged clip leaves the training slice and its base clip returns in its place."""
from __future__ import annotations


def narrator_overlap(event: dict, spans: list[tuple[float, float]], min_sec: float = 0.3) -> bool:
    """The extension (event start-end, episode time) overlaps transcribed narrator speech by at least min_sec."""
    a, b = event["start"], event["end"]
    return any(min(b, e) - max(a, s) >= min_sec for s, e in spans)


def who_conflict(who: str | None, series: str, line_speaker: str, clusters: dict[tuple[str, str], str]) -> bool:
    """The subtitle names the sound's maker and that name maps to another cluster than the line's speaker.
    Unknown names never conflict."""
    if not who:
        return False
    spk = clusters.get((series, who.strip()))
    return spk is not None and spk != line_speaker


def pitch_conflict(ext_f0: float | None, line_gender: str | None, male_line_max: float = 260.0,
                   female_line_min: float = 140.0) -> bool:
    """Male line with an extension median F0 >= male_line_max, or female line with <= female_line_min. Crying raises
    pitch, so the bars sit well outside voicegender's 150 / 175 Hz."""
    if ext_f0 is None:
        return False
    if line_gender == "male":
        return ext_f0 >= male_line_max
    if line_gender == "female":
        return ext_f0 <= female_line_min
    return False


CHECKS = ("narrator", "who_conflict", "pitch_conflict")


def flagged_ids(rows: list[dict]) -> set[str]:
    """Ids of event_checks.jsonl rows where any check is true."""
    return {r["id"] for r in rows if any(r.get(k) for k in CHECKS)}
