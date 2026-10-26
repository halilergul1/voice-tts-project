"""Verify event-extended clips: is the tagged sound really in the added audio, and is nobody speaking there?

Two measurements on the extension (the audio added by `ttsdata extend-events`):
- Whisper on the extension alone: laughter comes back as "Hahaha" or nothing; the audio-description narrator, who
  speaks in exactly these gaps of an accessible broadcast, comes back as words. (The packed re-check of the whole
  clip misses the narrator, because its words fall outside the aligned span.)
- A PANNs AudioSet tagger (Cnn14, frame-wise): probability of the tag's own classes (Laughter, Crying, Breathing,
  ...) and of speech classes (Speech, Narration, ...) over the extension.

Verdict: a sentence of real words, or 1-2 words plus speech probability >= event_speech_prob -> rejected
("speech_in_extension"); verified when the tag's AudioSet class reaches event_tag_prob or, for laughs, when Whisper
wrote at least two ha-like syllables; otherwise "unconfirmed": the extension is clean but nothing positive was
heard. On a TV mix the tagger mostly hears "Music" and never fires for crying or breathing, so cries and breaths
end up unconfirmed and need a human check (listen_verify/) or a better detector before training on them.
Measurements are cached per clip span (verify_cache.jsonl), so thresholds and token rules can change and be
re-applied without recomputation. Writes `verified` and `detector` into data/dataset_events/metadata.jsonl; the
csv files keep only verified rows.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from .config import Config
from .soundtags import tr_lower

TAG_CLASSES = {
    "laugh": ["Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle"],
    "cry": ["Crying, sobbing", "Baby cry, infant cry", "Whimper", "Wail, moan"],
    "breath": ["Breathing", "Wheeze", "Gasp", "Pant", "Sigh"],
    "sigh": ["Sigh", "Breathing", "Gasp"],
    "cough": ["Cough", "Throat clearing"],
    "throat_clear": ["Throat clearing", "Cough"],
    "scream": ["Screaming", "Shout", "Yell", "Whoop"],
    "shout": ["Shout", "Yell", "Screaming", "Whoop"],
    "moan": ["Groan", "Grunt", "Whimper", "Wail, moan"],
    "yawn": ["Sigh", "Breathing", "Groan"],  # AudioSet has no yawn class
    "sniff": ["Sniff", "Breathing"],
}
SPEECH_CLASSES = ["Speech", "Narration, monologue", "Female speech, woman speaking", "Male speech, man speaking",
                  "Conversation"]
_LAUGH_TOKEN = re.compile(r"^k?(?:h?[aeıioöuü]+h*)+$")  # ha, haha, hı hı, ahahah, kıhıhı (a cough / laugh with a k)
_LAUGH_SYL = re.compile(r"[kh]?[aeıioöuü]+h?")
INTERJECTIONS = {"ay", "oy", "of", "öf", "uf", "hmm", "hm", "hı", "hıh", "ee", "aa", "ya", "oha", "vay", "ah", "oh",
                 "eh", "ıh", "uh", "hey", "hişt", "şşş", "pff", "pöf", "öff", "tüh", "tsk", "cık", "ıı", "mm", "mmm",
                 "ahem", "ehem", "öhö", "öhöm", "khm", "hmph", "ahh", "ohh", "aah", "ooh"}
# Whisper's credits hallucination on non-speech audio ("Altyazı M.K."): dropped from the text, the rest still counts
HALLUCINATION_RE = re.compile(r"altyazı\w*(?:[\s:]+\S+){0,2}|abone ol\w*|kanalıma\b|takip edin|\bm\.k\.?", re.IGNORECASE)
_ANNOTATION_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]|\*[^*]*\*")


def speech_tokens(text: str) -> list[str]:
    """Real words Whisper heard in an extension: laughter syllables, interjections, annotations and the usual
    credits hallucinations removed."""
    if not text:
        return []
    t = tr_lower(_ANNOTATION_RE.sub(" ", HALLUCINATION_RE.sub(" ", text)))
    out = []
    for tok in re.findall(r"[\wçğıöşü']+", t):
        tok = tok.strip("'")
        if not tok or tok in INTERJECTIONS or _LAUGH_TOKEN.match(tok) or not re.search(r"[aeıioöuü]", tok):
            continue  # laughter, interjections and vowel-less fragments ("k" from a half-eaten "M.K.") are not words
        out.append(tok)
    return out


def laugh_syllables(text: str) -> int:
    """How many ha-like syllables Whisper wrote for the extension ("Hahahaha" -> 4, "Hı hı hı" -> 3)."""
    if not text:
        return 0
    n = 0
    for tok in re.findall(r"[\wçğıöşü]+", tr_lower(_ANNOTATION_RE.sub(" ", HALLUCINATION_RE.sub(" ", text)))):
        if _LAUGH_TOKEN.match(tok):  # "hı hı hı": an interjection once, laughter when it repeats
            n += len(_LAUGH_SYL.findall(tok))
    return n


def evidence(tag: str, words: list[str], speech_prob: float, tag_prob: float, thr: dict,
             laugh_syl: int = 0) -> str | None:
    """Why the tagged sound is believed to be in the extension: "laugh_syllables" (Whisper wrote ha-ha),
    "tagger" (the tag's AudioSet class fired), "clean" (no words and no speech-like audio: the subtitle cue stands,
    accepted when thr["accept_clean"], default True, since listening found the sound in every such sample), or None."""
    if tag == "laugh" and laugh_syl >= thr.get("laugh_syllables", 2):
        return "laugh_syllables"
    if tag_prob >= thr.get(f"tag_prob_{tag}", thr["tag_prob"]):
        return "tagger"
    if thr.get("accept_clean", True) and not words and speech_prob < thr["speech_prob"]:
        return "clean"
    return None


def verdict(tag: str, words: list[str], speech_prob: float, tag_prob: float, thr: dict,
            laugh_syl: int = 0) -> tuple[bool, str | None]:
    """(verified, reason). thr: speech_prob, tag_prob, optional tag_prob_<tag>, laugh_syllables (2), accept_clean (True).

    Speech in the extension: a sentence (3+ real words) always, 1-2 words when the tagger hears speech too.
    Otherwise verified when `evidence` finds something; "unconfirmed" when the tagger hears speech-like audio
    without words (sobbing with words, or a narrator Whisper missed) and nothing positive was heard."""
    if len(words) >= 3 or (words and speech_prob >= thr["speech_prob"]):
        return False, "speech_in_extension"
    if evidence(tag, words, speech_prob, tag_prob, thr, laugh_syl) is not None:
        return True, None
    return False, "unconfirmed"


# ----------------------------------------------------------------------------------------------- driver
def _resample(y: np.ndarray, sr: int, target: int) -> np.ndarray:
    from scipy.signal import resample_poly

    if sr == target:
        return y.astype(np.float32)
    g = gcd(sr, target)
    return resample_poly(y, target // g, sr // g).astype(np.float32)


def pad_to(y: np.ndarray, sr: int, min_sec: float) -> np.ndarray:
    """Zero-pad short audio symmetrically to min_sec (Cnn14 pools the time axis five times; < ~1 s collapses)."""
    need = int(round(min_sec * sr)) - len(y)
    if need <= 0:
        return y
    return np.pad(y, (need // 2, need - need // 2))


class _Tagger:
    def __init__(self, device: str):
        from panns_inference import SoundEventDetection, labels

        self.sed = SoundEventDetection(checkpoint_path=None, device=device)
        self.index = {name: k for k, name in enumerate(labels)}

    def probs(self, y32: np.ndarray, classes: list[str]) -> float:
        """Max over frames of the max over `classes`."""
        fw = self.sed.inference(y32[None, :])[0]  # (frames, 527)
        cols = [self.index[c] for c in classes if c in self.index]
        return float(fw[:, cols].max()) if cols and len(fw) else 0.0

    def frame_probs(self, y32: np.ndarray) -> np.ndarray:
        return self.sed.inference(y32[None, :])[0]


def _extension(y: np.ndarray, sr: int, row: dict, base: dict) -> np.ndarray:
    if row["edge"] == "start":
        return y[: int(round((base["orig_start"] - row["orig_start"]) * sr))]
    return y[int(round((base["orig_end"] - row["orig_start"]) * sr)):]


def _measure(row: dict, base: dict, wav: Path, cfg: Config, tagger: _Tagger) -> dict:
    from .asr import transcribe_words

    y, sr = sf.read(wav, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    ext = _extension(y, sr, row, base)
    if len(ext) < sr // 10:
        return {"asr_ext": "", "words": [], "speech_prob": 0.0, "tag_prob": 0.0, "ext_sec": round(len(ext) / sr, 2)}
    asr = transcribe_words(_resample(ext, sr, 16000), cfg)["asr_text"]
    fw = tagger.frame_probs(pad_to(_resample(ext, sr, 32000), 32000, 1.0))
    idx = tagger.index
    speech = float(fw[:, [idx[c] for c in SPEECH_CLASSES if c in idx]].max()) if len(fw) else 0.0
    tag_cols = [idx[c] for c in TAG_CLASSES.get(row["tag"], []) if c in idx]
    tag_p = float(fw[:, tag_cols].max()) if tag_cols and len(fw) else 0.0
    return {"asr_ext": asr, "words": speech_tokens(asr), "speech_prob": round(speech, 3), "tag_prob": round(tag_p, 3),
            "ext_sec": round(len(ext) / sr, 2)}


def run(cfg: Config, report: bool = True, limit: int = 0, out_dir: Path | None = None) -> dict:
    from .asr import resolve_backend

    out = out_dir or cfg.root / "dataset_events"
    meta = out / "metadata.jsonl"
    if not meta.exists():
        raise SystemExit(f"{meta} missing: run `ttsdata extend-events` first")
    rows = [json.loads(l) for l in meta.read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit:
        rows = rows[:limit]
    bases = {}
    for root in (cfg.dataset_dir, cfg.root / "dataset_long"):
        p = root / "metadata.jsonl"
        if p.exists():
            for l in p.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    r = json.loads(l)
                    bases[r["id"]] = r
    backend, model = resolve_backend(cfg)
    model_tag = f"{backend}:{model}|panns-cnn14"
    cache_file = out / "verify_cache.jsonl"
    cache: dict[str, dict] = {}
    if cache_file.exists():
        for l in cache_file.read_text(encoding="utf-8").splitlines():
            if l.strip():
                try:
                    c = json.loads(l)
                    cache[c["key"]] = c
                except json.JSONDecodeError:
                    continue

    def key(r):
        return f"{r['id']}|{r['orig_start']:.3f}|{r['orig_end']:.3f}|{model_tag}"

    todo = [r for r in rows if key(r) not in cache]
    if todo:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"  # PANNs on MPS is untested; CPU is fast enough
        tagger = _Tagger(device)
        with cache_file.open("a", encoding="utf-8") as f, tqdm(total=len(todo), desc="verifying", unit="clip") as bar:
            for r in todo:
                m = _measure(r, bases[r["base"]], out / "wavs" / f"{r['id']}.wav", cfg, tagger)
                m["key"] = key(r)
                cache[m["key"]] = m
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
                f.flush()
                bar.update(1)

    thr = {"speech_prob": cfg.event_speech_prob, "tag_prob": cfg.event_tag_prob, "accept_clean": cfg.event_accept_clean}
    reasons: Counter = Counter()
    for r in rows:
        m = cache[key(r)]
        words, syl = speech_tokens(m["asr_ext"]), laugh_syllables(m["asr_ext"])  # re-derived: the token rules may change
        ok, why = verdict(r["tag"], words, m["speech_prob"], m["tag_prob"], thr, syl)
        r["verified"] = ok
        r["detector"] = {"asr_ext": m["asr_ext"], "speech_words": words, "laugh_syllables": syl,
                         "speech_prob": m["speech_prob"], "tag_prob": m["tag_prob"], "ext_sec": m.get("ext_sec"),
                         "reason": why, "evidence": evidence(r["tag"], words, m["speech_prob"], m["tag_prob"], thr, syl) if ok else None,
                         "thresholds": thr}
        reasons[why or "verified"] += 1

    def lj(r):
        return f"{r['id']}|{r['text_tagged']}|{r['normalized_text_tagged']}|{r['speaker_id']}\n"

    tmp = meta.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fj:
        for r in rows:
            fj.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, meta)
    with (out / "metadata.csv").open("w", encoding="utf-8") as fa, (out / "train.csv").open("w", encoding="utf-8") as ft, \
         (out / "val.csv").open("w", encoding="utf-8") as fv:
        for r in rows:
            if r["verified"]:
                fa.write(lj(r))
                (fv if r.get("split") == "val" else ft).write(lj(r))

    per_tag = defaultdict(Counter)
    for r in rows:
        per_tag[r["tag"]][r["detector"]["reason"] or "verified"] += 1
    st = {"rows": len(rows), "verified": sum(1 for r in rows if r["verified"]), "reasons": dict(reasons.most_common()),
          "per_tag": {t: dict(c) for t, c in sorted(per_tag.items())}, "thresholds": thr, "model": model_tag}
    stats_file = out / "stats.json"
    stats = json.loads(stats_file.read_text(encoding="utf-8")) if stats_file.exists() else {}
    stats["verification"] = st
    stats_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_listen(out, rows)
    if report:
        print(report_table(rows))
    return st


def report_table(rows: list[dict]) -> str:
    lines = ["| tag | rows | verified (ha-ha / tagger / clean) | speech in extension | unconfirmed | tag_prob median | speech_prob median |",
             "|---|---|---|---|---|---|---|"]
    for tag in sorted({r["tag"] for r in rows}):
        rs = [r for r in rows if r["tag"] == tag]
        tp = np.array([r["detector"]["tag_prob"] for r in rs])
        sp = np.array([r["detector"]["speech_prob"] for r in rs])
        reasons = Counter(r["detector"]["reason"] for r in rs)
        ev = Counter(r["detector"].get("evidence") for r in rs if r["verified"])
        lines.append(f"| {tag} | {len(rs)} | {sum(ev.values())} ({ev.get('laugh_syllables', 0)} / {ev.get('tagger', 0)} / {ev.get('clean', 0)}) | "
                     f"{reasons.get('speech_in_extension', 0)} | {reasons.get('unconfirmed', 0)} | {np.median(tp):.2f} | {np.median(sp):.2f} |")
    return "\n".join(lines)


def _write_listen(out: Path, rows: list[dict], n: int = 5) -> None:
    import random

    d = out / "listen_verify"
    d.mkdir(exist_ok=True)
    for p in d.iterdir():
        if p.is_symlink():
            p.unlink()
    rng = random.Random(2)
    idx = ["# Verifier decisions to check by ear", "", "| file | decision | tag | Whisper heard | speech / tag prob | text |",
           "|---|---|---|---|---|---|"]
    k = 0
    groups = defaultdict(list)
    for r in rows:
        groups[r["detector"]["reason"] or f"verified_{r['detector'].get('evidence')}"].append(r)
    for reason, rs in sorted(groups.items()):
        for r in rng.sample(rs, min(n, len(rs))):
            name = f"{k:03d}_{reason}_{r['id']}.wav"
            os.symlink(os.path.relpath(out / "wavs" / f"{r['id']}.wav", d), d / name)
            dd = r["detector"]
            idx.append(f"| `{name}` | {reason} | {r['tag']} | {dd['asr_ext'][:40]!r} | {dd['speech_prob']:.2f} / {dd['tag_prob']:.2f} | {r['text_tagged']} |")
            k += 1
    (d / "index.md").write_text("\n".join(idx) + "\n", encoding="utf-8")
