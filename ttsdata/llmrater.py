"""An audio-input LLM as a blind emotion rater, graded against the human ratings.

Gemini (generateContent REST; Google calls it legacy but keeps it fully supported) hears a calibration clip without
its transcript and returns JSON: a coarse emotion, a self-reported confidence and a few words on the delivery. The
transcript is withheld on purpose: audio LLMs are known to label from the words (LISTEN, ACL 2026), and a TTS style
label must describe the voice. Everything here is pure; the HTTP calls live in scripts/llm_rater.py, which reads the
API key from the environment or a file and never prints it.
"""
from __future__ import annotations

import base64
import json
import re

from .emotion import COARSE, CURVE_THRESHOLDS, precision_curve, score_ratings

LABELS = ("angry", "happy", "sad", "neutral", "other")
PROMPT = (
    "You hear a short clip of Turkish speech taken from a TV drama. Judge the speaker's emotional state from the "
    "VOICE ALONE: tone, pitch, loudness, tempo, voice quality. Do not use the meaning of the words. "
    "Pick exactly one of: angry (anger, irritation, contempt), happy (joy, amusement, warmth), sad (sadness, grief, "
    "disappointment), neutral (calm or matter-of-fact delivery, no clear emotion), other (fear, surprise, disgust, "
    "mixed, or more than one speaker). Give a confidence between 0 and 1 for that choice, and describe the delivery "
    "in at most six English words (for example: shouting, whispering, tearful, sarcastic, pleading, calm narration)."
)
SCHEMA = {
    "type": "OBJECT",
    "properties": {"emotion": {"type": "STRING", "enum": list(LABELS)}, "confidence": {"type": "NUMBER"},
                   "style": {"type": "STRING"}},
    "required": ["emotion", "confidence", "style"],
}


def request_body(audio: bytes, mime: str) -> dict:
    """generateContent body: the prompt and the clip, JSON output pinned to SCHEMA, temperature 0. No transcript."""
    return {"contents": [{"role": "user", "parts": [
                {"text": PROMPT},
                {"inline_data": {"mime_type": mime, "data": base64.b64encode(audio).decode()}}]}],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": SCHEMA, "temperature": 0}}


def parse_response(resp: dict) -> dict:
    """Model reply -> label (one of LABELS, or None when it gave no usable JSON), confidence clamped to 0-1, style,
    finish reason, token count and the raw text. An emotion outside the four coarse classes counts as "other"."""
    cands = resp.get("candidates") or []
    finish = cands[0].get("finishReason") if cands else (resp.get("promptFeedback") or {}).get("blockReason")
    text = "".join(p.get("text", "") for c in cands[:1] for p in (c.get("content") or {}).get("parts", []))
    out = {"label": None, "conf": None, "style": "", "finish": finish,
           "tokens": (resp.get("usageMetadata") or {}).get("totalTokenCount"), "raw": text}
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return out
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return out
    lab = str(d.get("emotion", "")).strip().lower()
    out["label"] = lab if lab in COARSE else "other"
    try:
        conf = float(d.get("confidence"))
    except (TypeError, ValueError):
        conf = None
    out["conf"] = None if conf is None else round(min(1.0, max(0.0, conf)), 3)
    out["style"] = str(d.get("style") or "").strip()
    return out


def preds_for_scoring(rows: list[dict]) -> dict[str, tuple[str | None, float]]:
    """Cached rater rows -> the {id: (label, conf)} shape score_ratings takes; "other" and failures abstain."""
    return {r["id"]: (r["label"] if r.get("label") in COARSE else None, float(r.get("conf") or 0.0)) for r in rows}


def backoff_seconds(attempt: int, retry_after: float | None = None) -> float:
    """Wait before retrying a 429 / 5xx: the server's Retry-After when given, else 5 s doubling, capped at 60 s."""
    return retry_after if retry_after is not None else float(min(60, 5 * 2 ** attempt))


class Pacer:
    """Spaces request starts at most `rpm` per minute across threads: wait() blocks until a slot is free and
    returns the seconds it slept."""

    def __init__(self, rpm: float, clock=None, sleep=None, max_gap: float = 60.0):
        import threading
        import time

        self.min_gap = self.gap = 60.0 / rpm
        self.max_gap = max_gap
        self.clock = clock or time.monotonic
        self.sleep = sleep or time.sleep
        self.lock = threading.Lock()
        self.next_at = self.clock()

    def wait(self) -> float:
        with self.lock:
            now = self.clock()
            delay = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + self.gap
        if delay:
            self.sleep(delay)
        return delay

    def slow_down(self) -> float:
        """After a 429: double the gap between starts, up to max_gap."""
        with self.lock:
            self.gap = min(self.max_gap, self.gap * 2)
            return self.gap

    def recover(self) -> None:
        """After a success: ease the gap back towards the configured rate."""
        with self.lock:
            self.gap = max(self.min_gap, self.gap * 0.9)


def quota_kind(err: dict) -> str:
    """Which quota a 429 names: "day" (stop, resume tomorrow), "minute" (slow down) or "unknown"."""
    e = err.get("error") or {}
    text = " ".join([str(e.get("message", ""))] + [f"{v.get('quotaMetric', '')} {v.get('quotaId', '')}"
                                                      for d in e.get("details", []) for v in d.get("violations", [])]).lower()
    if "perday" in text or "per day" in text or "per_day" in text or "daily" in text:
        return "day"
    if "perminute" in text or "per minute" in text or "per_minute" in text:
        return "minute"
    return "unknown"


def report(model: str, llm_rows: list[dict], e2v: dict[str, tuple[str | None, float]], ratings: dict[str, str],
           cost_tokens: int | None = None) -> str:
    """Markdown: the LLM against the ratings, emotion2vec+ on the same clips, how often the two agree, and the
    delivery words the LLM used (the raw material for a free-text style caption)."""
    preds = preds_for_scoring(llm_rows)
    m = score_ratings(preds, ratings)
    m2 = score_ratings({k: v for k, v in e2v.items() if k in preds}, ratings)
    curve = precision_curve(preds, ratings)
    both = [(preds[i][0], e2v[i][0]) for i in preds if i in e2v and preds[i][0] and e2v[i][0]]
    agree = sum(1 for a, b in both if a == b)
    lines = [f"# LLM rater: {model}", "",
             f"{m['n']} rated clips heard without their transcript; labelled {m['labelled']} (the rest: other / no answer); "
             f"accuracy on the labelled {m['accuracy']}." + (f" Tokens used: {cost_tokens}." if cost_tokens else ""), ""]

    def table(mm):
        rows = ["| class | predicted | correct | precision | rated | recall |", "|---|---|---|---|---|---|"]
        rows += [f"| {c} | {d['predicted']} | {d['correct']} | {d['precision']} | {d['rated']} | {d['recall']} |"
                 for c, d in mm["per_class"].items()]
        return rows

    lines += [f"## {model}", ""] + table(m) + ["",
              "Precision when only answers at or above a self-reported confidence are kept (correct / kept):", "",
              "| class | " + " | ".join(f">= {t}" for t in CURVE_THRESHOLDS) + " |", "|---|" + "---|" * len(CURVE_THRESHOLDS)]
    for c in ("angry", "sad", "happy"):
        cells = {d["threshold"]: f"{d['precision']:.2f} ({d['correct']}/{d['kept']})" for d in curve.get(c, [])}
        lines.append(f"| {c} | " + " | ".join(cells.get(t, "-") for t in CURVE_THRESHOLDS) + " |")
    lines += ["", "Confusion (predicted -> rated): " + ", ".join(f"{k} {n}" for k, n in sorted(m["confusion"].items())), "",
              "## emotion2vec+ on the same clips (default cut)", ""] + table(m2) + ["",
              f"## Agreement: the two labelled the same clip {len(both)} times and agreed on {agree}"
              + (f" ({agree / len(both):.0%})" if both else ""), "",
              "## Delivery words from the LLM (first 40)", ""]
    lines += [f"- {r['id']}: {r.get('style') or '-'} ({r.get('label')}, rated {ratings.get(r['id'], '?')})"
              for r in llm_rows[:40]]
    return "\n".join(lines) + "\n"
