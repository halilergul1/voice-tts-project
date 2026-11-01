"""Whisper on the gaps between subtitle lines of Engelsiz episodes: the audio-description narrator (stage 2 Task 3B).

  .venv/bin/python scripts/transcribe_narration.py --source trt [--limit-episodes 20] [--device cuda]

Per episode (data/raw/<source>/<uid>/ with audio.* and subs.vtt / subs.srt): the speech cues (subtitles.clean_cue), the
gaps between them (narration.narration_gaps), all gaps of the episode joined with 1 s of silence and transcribed in one
batched faster-whisper pass (Turkish, word timestamps, VAD), the words handed back to their gap
(asr.split_pack_result) and grouped into spans (narration.spans_from_words). Appends {uid, start, end, text} in episode
seconds to <out> and the uid to <out stem>_episodes.txt (resumable per episode); per-episode gap seconds, narration
seconds and wall time go to <out stem>_stats.jsonl.
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
from ttsdata.asr import _preload_cuda_libs, split_pack_result  # noqa: E402
from ttsdata.narration import narration_gaps, spans_from_words  # noqa: E402
from ttsdata.restoration import episode_audio  # noqa: E402
from ttsdata.subtitles import clean_cue, parse  # noqa: E402

SR = 16_000
SILENCE = 1.0  # seconds between joined gaps


def _decode(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def _subs(ep: Path) -> Path | None:
    return next((p for p in sorted(ep.glob("subs.*")) if p.suffix in (".vtt", ".srt")), None)


def episode_spans(ep: Path, pipe, a) -> tuple[list[dict], dict]:
    y = _decode(episode_audio(ep))
    total = len(y) / SR
    cues = [(c.start, c.end) for c in parse(_subs(ep)) if clean_cue(c) is not None]
    gaps = narration_gaps(cues, total, min_gap=a.min_gap)[: a.max_gaps or None]
    stats = {"uid": ep.name, "audio_sec": round(total, 1), "gaps": len(gaps), "gap_sec": round(sum(b - s for s, b in gaps), 1)}
    if not gaps:
        return [], {**stats, "spans": 0, "narration_sec": 0.0}
    pieces, offsets, durs, t = [], [], [], 0.0
    sil = np.zeros(int(SILENCE * SR), dtype=np.float32)
    for s, e in gaps:
        seg = y[int(s * SR): int(e * SR)]
        offsets.append(t)
        durs.append(len(seg) / SR)
        pieces += [seg, sil]
        t += len(seg) / SR + SILENCE
    segments, _ = pipe.transcribe(np.concatenate(pieces), language="tr", task="transcribe", word_timestamps=True,
                                  batch_size=a.batch, beam_size=5, vad_filter=True)
    words = [{"word": w.word, "start": float(w.start), "end": float(w.end)} for s in segments for w in (s.words or [])]
    parts = split_pack_result({"words": words, "segments": []}, offsets, durs, slack=SILENCE / 2)
    rows = []
    for (gs, _), part in zip(gaps, parts):
        ws = [{**w, "start": gs + w["start"], "end": gs + w["end"]} for w in part["words"]]
        rows += [{"uid": ep.name, **sp} for sp in spans_from_words(ws, max_pause=a.max_pause)]
    return rows, {**stats, "spans": len(rows), "narration_sec": round(sum(r["end"] - r["start"] for r in rows), 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="trt")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/dataset/narration.jsonl")
    ap.add_argument("--limit-episodes", type=int, default=0, help="stop after N newly transcribed episodes")
    ap.add_argument("--min-gap", type=float, default=1.5)
    ap.add_argument("--max-pause", type=float, default=1.5, help="a longer pause between words starts a new span")
    ap.add_argument("--max-gaps", type=int, default=0, help="first N gaps per episode only (testing)")
    ap.add_argument("--model", default="deepdml/faster-whisper-large-v3-turbo-ct2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    from faster_whisper import BatchedInferencePipeline, WhisperModel

    out = Path(a.out)
    done_file = out.with_name(out.stem + "_episodes.txt")
    stats_file = out.with_name(out.stem + "_stats.jsonl")
    done = set(done_file.read_text(encoding="utf-8").split()) if done_file.exists() else set()
    eps = [p for p in sorted((Path(a.raw) / a.source).iterdir())
           if p.is_dir() and p.name not in done and episode_audio(p) and _subs(p)]
    if a.limit_episodes:
        eps = eps[: a.limit_episodes]
    print(f"{len(eps)} episodes to transcribe ({len(done)} done)", flush=True)
    if a.device == "cuda":
        _preload_cuda_libs()
    model = WhisperModel(a.model, device=a.device, compute_type="float16" if a.device == "cuda" else "int8")
    pipe = BatchedInferencePipeline(model)
    for i, ep in enumerate(eps, 1):
        t0 = time.time()
        rows, stats = episode_spans(ep, pipe, a)
        stats["wall_sec"] = round(time.time() - t0, 1)
        with out.open("a", encoding="utf-8") as f:
            f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        with stats_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(stats) + "\n")
        with done_file.open("a", encoding="utf-8") as f:
            f.write(ep.name + "\n")
        print(f"[{i}/{len(eps)}] {ep.name}: {stats['gaps']} gaps {stats['gap_sec']} s, {stats['spans']} spans "
              f"{stats['narration_sec']} s, {stats['wall_sec']} s wall", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
