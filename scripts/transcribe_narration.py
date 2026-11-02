"""Seamless M4T v2 on the gaps between subtitle lines of Engelsiz episodes: the audio-description narrator (stage 2
Task 3B).

  .venv/bin/python scripts/transcribe_narration.py --source trt [--limit-episodes 20] [--device cuda]

Not Whisper: Whisper skips the describer (pilot 2026-11-02: nothing, or the credit line "Bu dizinin betimlemesi...",
where the listener hears narration; it was trained on subtitles, which never contain the description). Per episode
(data/raw/<source>/<uid>/ with audio.* and subs.vtt / subs.srt): the speech cues (subtitles.clean_cue), the gaps
between them (narration.narration_gaps), Silero VAD speech inside each gap, grouped into chunks of at most 10 s
(narration.speech_chunks; Seamless drops words on longer inputs), each chunk transcribed in Turkish and cleaned
(narration.clean_span_text). Appends {uid, start, end, text} in episode seconds to <out> and the uid to
<out stem>_episodes.txt (resumable per episode); per-episode gap seconds, narration seconds and wall time go to
<out stem>_stats.jsonl.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from second_asr import Recogniser  # noqa: E402
from ttsdata.narration import clean_span_text, narration_gaps, speech_chunks  # noqa: E402
from ttsdata.restoration import episode_audio  # noqa: E402
from ttsdata.subtitles import clean_cue, parse  # noqa: E402

SR = 16_000


def _decode(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def _subs(ep: Path) -> Path | None:
    return next((p for p in sorted(ep.glob("subs.*")) if p.suffix in (".vtt", ".srt")), None)


def gap_speech(y: np.ndarray, gaps: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Silero VAD speech segments inside the gaps, in episode seconds."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(min_silence_duration_ms=500, speech_pad_ms=200, min_speech_duration_ms=300)
    out = []
    for gs, ge in gaps:
        for t in get_speech_timestamps(y[int(gs * SR): int(ge * SR)], opts, sampling_rate=SR):
            out.append((gs + t["start"] / SR, min(ge, gs + t["end"] / SR)))
    return out


def episode_spans(ep: Path, asr, a) -> tuple[list[dict], dict]:
    y = _decode(episode_audio(ep))
    total = len(y) / SR
    cues = [(c.start, c.end) for c in parse(_subs(ep)) if clean_cue(c) is not None]
    gaps = narration_gaps(cues, total, min_gap=a.min_gap)[: a.max_gaps or None]
    stats = {"uid": ep.name, "audio_sec": round(total, 1), "gaps": len(gaps), "gap_sec": round(sum(b - s for s, b in gaps), 1)}
    chunks = speech_chunks(gap_speech(y, gaps), max_pause=a.max_pause, max_len=a.max_len)
    stats["speech_sec"] = round(sum(e - s for s, e in chunks), 1)
    rows = []
    order = sorted(range(len(chunks)), key=lambda k: chunks[k][1] - chunks[k][0])  # similar lengths per batch
    texts: dict[int, str] = {}
    for i in range(0, len(order), a.batch):
        idx = order[i: i + a.batch]
        for k, t in zip(idx, asr([y[int(chunks[k][0] * SR): int(chunks[k][1] * SR)] for k in idx])):
            texts[k] = t
    for k, (s, e) in enumerate(chunks):
        text = clean_span_text(texts.get(k, ""))
        if text:
            rows.append({"uid": ep.name, "start": round(s, 3), "end": round(e, 3), "text": text})
    return rows, {**stats, "spans": len(rows), "narration_sec": round(sum(r["end"] - r["start"] for r in rows), 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="trt")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/dataset/narration.jsonl")
    ap.add_argument("--limit-episodes", type=int, default=0, help="stop after N newly transcribed episodes")
    ap.add_argument("--min-gap", type=float, default=1.5)
    ap.add_argument("--max-pause", type=float, default=1.0, help="speech closer than this joins one chunk")
    ap.add_argument("--max-len", type=float, default=10.0, help="longest chunk handed to the recogniser (s)")
    ap.add_argument("--max-gaps", type=int, default=0, help="first N gaps per episode only (testing)")
    ap.add_argument("--model", default="seamless", help="second_asr.Recogniser name")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    out = Path(a.out)
    done_file = out.with_name(out.stem + "_episodes.txt")
    stats_file = out.with_name(out.stem + "_stats.jsonl")
    done = set(done_file.read_text(encoding="utf-8").split()) if done_file.exists() else set()
    eps = [p for p in sorted((Path(a.raw) / a.source).iterdir())
           if p.is_dir() and p.name not in done and episode_audio(p) and _subs(p)]
    if a.limit_episodes:
        eps = eps[: a.limit_episodes]
    print(f"{len(eps)} episodes to transcribe ({len(done)} done)", flush=True)
    asr = Recogniser(a.model, device=a.device)
    for i, ep in enumerate(eps, 1):
        t0 = time.time()
        rows, stats = episode_spans(ep, asr, a)
        stats["wall_sec"] = round(time.time() - t0, 1)
        with out.open("a", encoding="utf-8") as f:
            f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        with stats_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(stats) + "\n")
        with done_file.open("a", encoding="utf-8") as f:
            f.write(ep.name + "\n")
        print(f"[{i}/{len(eps)}] {ep.name}: {stats['gaps']} gaps {stats['gap_sec']} s, {stats['spans']} spans "
              f"{stats['narration_sec']} s (VAD speech {stats['speech_sec']} s), {stats['wall_sec']} s wall", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
