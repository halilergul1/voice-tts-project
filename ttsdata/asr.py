"""Verify and trim each window with Whisper.

For every candidate window we transcribe with word timestamps, align the Whisper words to the
subtitle tokens, and keep exactly the aligned span. That gives (a) tight utterance boundaries that
exclude neighbouring speech and audio-description narration, and (b) a CER between subtitle and
transcript measured on the span only, so edge bleed does not cause false rejections.
"""
from __future__ import annotations

import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path

import jiwer
import numpy as np
from tqdm import tqdm

from .config import Config
from .locks import try_lock as _try_lock
from .locks import unlock as _unlock
from .textnorm import for_cer


def speech_cover(r: dict) -> float:
    """VAD speech seconds found in the window, relative to the subtitle cue length."""
    cue_dur = max(r["cue_end"] - r["cue_start"], 0.1)
    return r.get("speech_ratio", 0.0) * r.get("window_dur", 0.0) / cue_dur


_MLX_MODEL = "mlx-community/whisper-large-v3-turbo"
_CT2_MODEL = "deepdml/faster-whisper-large-v3-turbo-ct2"
_ct2 = None


def resolve_backend(cfg: Config) -> tuple[str, str]:
    backend = cfg.whisper_backend
    if backend == "auto":
        try:
            import mlx_whisper  # noqa: F401
            backend = "mlx"
        except ImportError:
            backend = "faster"
    model = cfg.whisper_model
    if model == "auto":
        model = _MLX_MODEL if backend == "mlx" else _CT2_MODEL
    return backend, model


def _words_from_mlx(audio, model: str) -> tuple[str, list[dict], list[dict]]:
    import mlx_whisper

    r = mlx_whisper.transcribe(audio, path_or_hf_repo=model, language="tr", task="transcribe",
                               word_timestamps=True, temperature=(0.0, 0.2, 0.4), condition_on_previous_text=False,
                               compression_ratio_threshold=2.4, no_speech_threshold=0.6, verbose=None)
    segs = [{"start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0)),
             "no_speech_prob": float(s.get("no_speech_prob", 0.0)), "avg_logprob": float(s.get("avg_logprob", -10.0))}
            for s in (r.get("segments") or [])]
    words = [w for s in (r.get("segments") or []) for w in (s.get("words") or [])]
    return r.get("text", ""), words, segs


def _load_16k(path: str) -> np.ndarray:
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly

    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        g = gcd(16000, sr)
        y = resample_poly(y, 16000 // g, sr // g).astype(np.float32)
    return np.ascontiguousarray(y, dtype=np.float32)


def _preload_cuda_libs() -> None:
    """CTranslate2 dlopens libcublas/libcudnn by soname; the pip wheels (nvidia-cublas-cu12, nvidia-cudnn-cu12)
    put them under site-packages/nvidia/*/lib, off the default loader path. Loading them by full path first
    makes the later lookups succeed. No-op when the packages are absent (macOS, CPU-only)."""
    import ctypes
    import glob
    import site

    roots = {p for p in site.getsitepackages() + [site.getusersitepackages()] if p}
    for pattern in ("libcublasLt.so.*", "libcublas.so.*", "libcudnn.so.*"):  # wheel layouts: nvidia/<lib>/lib or nvidia/cu12/lib
        for root in roots:
            for lib in sorted(glob.glob(f"{root}/nvidia/*/lib/{pattern}")):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def _words_from_ct2(audio, model: str, cfg: Config) -> tuple[str, list[dict], list[dict]]:
    global _ct2
    from faster_whisper import WhisperModel

    if _ct2 is None:
        _preload_cuda_libs()
        device = cfg.whisper_device
        if device == "auto":
            try:
                import ctranslate2
                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                device = "cpu"
        compute = cfg.whisper_compute
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        _ct2 = WhisperModel(model, device=device, compute_type=compute)
    if isinstance(audio, str):
        audio = _load_16k(audio)  # bypass faster-whisper's PyAV loader (version-fragile)
    seg_iter, _info = _ct2.transcribe(audio, language="tr", task="transcribe", word_timestamps=True, beam_size=5,
                                      temperature=[0.0, 0.2, 0.4], condition_on_previous_text=False,
                                      compression_ratio_threshold=2.4, no_speech_threshold=0.6, vad_filter=False)
    segs, words, text = [], [], []
    for s in seg_iter:
        segs.append({"start": float(s.start), "end": float(s.end), "no_speech_prob": float(s.no_speech_prob),
                     "avg_logprob": float(s.avg_logprob)})
        text.append(s.text)
        for w in s.words or []:
            words.append({"word": w.word, "start": w.start, "end": w.end, "probability": w.probability})
    return "".join(text), words, segs


def transcribe_words(audio, cfg: Config) -> dict:
    """Whisper with word timestamps on a file path or a 16 kHz float32 array."""
    backend, model = resolve_backend(cfg)
    if isinstance(audio, Path):
        audio = str(audio)
    if backend == "mlx":
        text, raw_words, segs = _words_from_mlx(audio, model)
    else:
        text, raw_words, segs = _words_from_ct2(audio, model, cfg)
    words = []
    for w in raw_words:
        tok = for_cer(w["word"])
        if tok:
            words.append({"word": tok, "start": float(w["start"]), "end": float(w["end"]),
                          "prob": float(w.get("probability", 1.0))})
    return {
        "asr_text": text.strip(),
        "words": words,
        "segments": segs,
        "no_speech_prob": float(np.mean([s["no_speech_prob"] for s in segs])) if segs else 1.0,
        "avg_logprob": float(np.mean([s["avg_logprob"] for s in segs])) if segs else -10.0,
    }


# --------------------------------------------------------------------------- packing
def pack_records(recs: list[dict], pack_seconds: float, gap: float, seed: int) -> list[list[dict]]:
    """Group clips so each pack fits one Whisper window. Order is shuffled so that overlapping
    neighbouring windows from the same episode do not sit next to each other in a pack."""
    import random

    recs = list(recs)
    random.Random(seed).shuffle(recs)
    packs, cur, used = [], [], 0.0
    for r in recs:
        d = r["window_dur"] + gap
        if cur and used + d > pack_seconds:
            packs.append(cur)
            cur, used = [], 0.0
        cur.append(r)
        used += d
    if cur:
        packs.append(cur)
    return packs


def split_pack_result(asr: dict, offsets: list[float], durations: list[float], slack: float) -> list[dict]:
    """Hand the words/segments of a packed transcription back to each clip, in clip-relative time.

    Each word goes to the clip nearest to its midpoint (distance 0 when inside the clip); words farther
    than `slack` from every clip are hallucinations in the silence gaps and are dropped.
    """
    buckets: list[list[dict]] = [[] for _ in offsets]
    for w in asr["words"]:
        mid = (w["start"] + w["end"]) / 2
        best, best_d = None, None
        for k, (off, dur) in enumerate(zip(offsets, durations)):
            d = 0.0 if off <= mid <= off + dur else min(abs(mid - off), abs(mid - (off + dur)))
            if best_d is None or d < best_d:
                best, best_d = k, d
        if best is not None and best_d <= slack:
            off, dur = offsets[best], durations[best]
            buckets[best].append({**w, "start": min(max(w["start"] - off, 0.0), dur), "end": min(max(w["end"] - off, 0.0), dur)})
    out = []
    for words, off, dur in zip(buckets, offsets, durations):
        segs = [s for s in asr["segments"] if s["end"] > off and s["start"] < off + dur]
        out.append({
            "asr_text": " ".join(w["word"] for w in words),
            "words": words,
            "segments": segs,
            "no_speech_prob": float(np.mean([s["no_speech_prob"] for s in segs])) if segs else 1.0,
            "avg_logprob": float(np.mean([s["avg_logprob"] for s in segs])) if segs else -10.0,
        })
    return out


def verify_pack(pack: list[dict], cfg: Config) -> list[dict]:
    audios = [_load_16k(r["wav"]) for r in pack]
    gap = np.zeros(int(cfg.pack_gap * 16000), dtype=np.float32)
    offsets, pieces, t = [], [], 0.0
    for a in audios:
        offsets.append(t)
        pieces += [a, gap]
        t += len(a) / 16000 + cfg.pack_gap
    asr = transcribe_words(np.concatenate(pieces), cfg)
    parts = split_pack_result(asr, offsets, [len(a) / 16000 for a in audios], cfg.pack_slack)
    results = []
    for r, part in zip(pack, parts):
        res = evaluate(r, part, cfg)
        res["id"] = r["id"]
        res["packed"] = len(pack)
        results.append(res)
    return results


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def align(ref_tokens: list[str], hyp_words: list[dict]) -> tuple[int, int, int, bool, bool] | None:
    """Token-level edit-distance alignment with fuzzy substitution cost.

    Returns (first_hyp_idx, last_hyp_idx, n_aligned, first_ref_matched, last_ref_matched) for the
    hyp span covering the reference, or None when nothing aligns.
    """
    n, m = len(ref_tokens), len(hyp_words)
    if n == 0 or m == 0:
        return None
    INS, DEL = 0.5, 1.0  # extra ASR words inside the span are cheap (ASR splits words); missing ref words are not
    D = np.zeros((n + 1, m + 1))
    D[:, 0] = np.arange(n + 1) * DEL
    D[0, :] = 0.0  # free skipping of leading hyp words (they belong to the neighbour)
    back = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0=sub/match, 1=del(ref), 2=ins(hyp)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sim = _sim(ref_tokens[i - 1], hyp_words[j - 1]["word"])
            sub = D[i - 1, j - 1] + ((1.0 - sim) if sim >= 0.5 else 1.0)
            dele = D[i - 1, j] + DEL
            ins = D[i, j - 1] + INS
            best = min(sub, dele, ins)
            D[i, j] = best
            back[i, j] = 0 if best == sub else (1 if best == dele else 2)
    j = int(np.argmin(D[n, :]))  # free skipping of trailing hyp words
    i = n
    matched: list[int] = []
    matched_ref: set[int] = set()
    while i > 0 and j > 0:
        op = back[i, j]
        if op == 0:
            if _sim(ref_tokens[i - 1], hyp_words[j - 1]["word"]) >= 0.5:
                matched.append(j - 1)
                matched_ref.add(i - 1)
            i -= 1
            j -= 1
        elif op == 1:
            i -= 1
        else:
            j -= 1
    if not matched:
        return None
    return min(matched), max(matched), len(matched), 0 in matched_ref, (n - 1) in matched_ref


def trim_bounds(span_start: float, span_end: float, prev_end: float, next_start: float, first_ok: bool,
                last_ok: bool, window_dur: float, cfg: Config) -> tuple[float, float]:
    """Clip edges: aligned span plus a margin, never past the neighbouring Whisper word.

    When the subtitle's first/last token has no Whisper word (interjections, swallowed words), its audio
    sits in the gap before/after the span, so the whole gap is kept instead of cutting the word off.
    """
    t0 = prev_end if not first_ok else max(prev_end, span_start - cfg.trim_margin_before, 0.0)
    t1 = next_start if not last_ok else min(next_start, span_end + cfg.trim_margin_after, window_dur)
    return t0, t1


def evaluate(rec: dict, asr: dict, cfg: Config) -> dict:
    ref_norm = for_cer(rec["text"])
    ref_tokens = ref_norm.split()
    words = asr["words"]
    out = {"asr_text": asr["asr_text"], "no_speech_prob": round(asr["no_speech_prob"], 4),
           "avg_logprob": round(asr["avg_logprob"], 4)}
    span = align(ref_tokens, words)
    if span is None:
        out.update(cer=1.0, aligned=0, span_text="", trim_start=None, trim_end=None)
        return out
    j0, j1, n_al, first_ok, last_ok = span
    span_words = words[j0:j1 + 1]
    hyp = " ".join(w["word"] for w in span_words)
    cer = jiwer.cer(ref_norm, hyp) if ref_norm and hyp else 1.0
    window_dur = rec["window_dur"]
    prev_end = words[j0 - 1]["end"] if j0 > 0 else 0.0
    next_start = words[j1 + 1]["start"] if j1 + 1 < len(words) else window_dur
    t0, t1 = trim_bounds(span_words[0]["start"], span_words[-1]["end"], prev_end, next_start, first_ok, last_ok,
                         window_dur, cfg)
    out.update(
        cer=round(float(cer), 4), aligned=n_al, ref_tokens=len(ref_tokens), span_text=hyp,
        edge_unmatched=(not first_ok) or (not last_ok),
        word_prob=round(float(np.mean([w["prob"] for w in span_words])), 3),
        bleed_before=j0, bleed_after=len(words) - 1 - j1,
        span_start=round(span_words[0]["start"], 3), span_end=round(span_words[-1]["end"], 3),
        prev_end=round(prev_end, 3), next_start=round(next_start, 3), first_ok=first_ok, last_ok=last_ok,
        trim_start=round(t0, 3), trim_end=round(t1, 3), duration=round(t1 - t0, 3),
    )
    return out


def try_lock(seg_dir: Path) -> bool:
    return _try_lock(seg_dir, ".verify.lock")


def unlock(seg_dir: Path) -> None:
    _unlock(seg_dir, ".verify.lock")


def verify_episode(seg_dir: Path, cfg: Config, candidates_only: bool = True, asr_filename: str = "asr.jsonl") -> dict:
    """Append ASR results to <seg_dir>/<asr_filename>; resumable (skips ids already present)."""
    seg_file = seg_dir / "segments.jsonl"
    asr_file = seg_dir / asr_filename
    done = set()
    if asr_file.exists():
        for line in asr_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["id"])
                except json.JSONDecodeError:
                    pass
    todo = []
    for line in seg_file.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["id"] in done or "wav" not in r:
            continue
        cue_dur = r["cue_end"] - r["cue_start"]
        if candidates_only and not (cfg.min_dur - 0.4 <= cue_dur <= cfg.max_dur + 0.5):
            continue
        if candidates_only and speech_cover(r) < cfg.min_speech_ratio:
            continue
        todo.append(r)
    if not todo:
        return {"uid": seg_dir.name, "already": len(done), "transcribed": 0}
    if not try_lock(seg_dir):
        return {"uid": seg_dir.name, "skipped": "locked by another worker"}

    def one_by_one(recs):
        for r in recs:
            try:
                res = evaluate(r, transcribe_words(r["wav"], cfg), cfg)
            except Exception as e:  # keep going; a crash on one clip must not kill the run
                res = {"asr_text": "", "cer": 1.0, "trim_start": None, "trim_end": None, "error": str(e)[:200]}
            res["id"] = r["id"]
            yield res

    def packed(recs):
        for pack in pack_records(recs, cfg.pack_seconds, cfg.pack_gap, cfg.seed):
            try:
                yield from verify_pack(pack, cfg)
            except Exception:
                yield from one_by_one(pack)  # fall back for this pack only

    n_ok = 0
    try:
        with asr_file.open("a", encoding="utf-8") as f:
            stream = packed(todo) if cfg.pack_seconds > 0 else one_by_one(todo)
            for res in tqdm(stream, total=len(todo), desc=seg_dir.name, unit="seg", leave=False, mininterval=5.0):
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                f.flush()
                n_ok += 1
    finally:
        unlock(seg_dir)
    return {"uid": seg_dir.name, "already": len(done), "transcribed": n_ok}
