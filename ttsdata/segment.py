"""Cut subtitle-aligned speech segments out of an episode's audio, trimmed by VAD."""
from __future__ import annotations

import json
import os
import subprocess
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import Config
from .subtitles import Segment, load_segments

_vad = None


def decode_audio(path: Path, sr: int) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def _vad_model():
    global _vad
    if _vad is None:
        import torch
        from silero_vad import load_silero_vad
        torch.set_num_threads(int(os.environ.get("TTSDATA_TORCH_THREADS", "2")))  # several segmenters share the CPU
        _vad = load_silero_vad()
    return _vad


def speech_regions(x: np.ndarray, sr: int, threshold: float) -> list[tuple[float, float]]:
    import torch
    from silero_vad import get_speech_timestamps

    if sr != 16000:
        g = gcd(16000, sr)
        x16 = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    else:
        x16 = x
    ts = get_speech_timestamps(torch.from_numpy(x16), _vad_model(), sampling_rate=16000, threshold=threshold,
                               min_speech_duration_ms=100, min_silence_duration_ms=100, speech_pad_ms=30,
                               return_seconds=True)
    return [(float(t["start"]), float(t["end"])) for t in ts]


def cut_episode(raw_dir: Path, cfg: Config) -> dict:
    info = json.loads((raw_dir / "info.json").read_text(encoding="utf-8"))
    uid, source = info["uid"], info["source"]
    audio_path = next(raw_dir.glob("audio.*"))
    subs_path = next(raw_dir.glob("subs.*"))
    out_dir = cfg.seg_dir / source / uid
    out_dir.mkdir(parents=True, exist_ok=True)

    segs, sub_stats = load_segments(subs_path, cfg.merge_gap, cfg.merge_gap_comma, cfg.max_merged_dur)
    audio = decode_audio(audio_path, cfg.sample_rate)
    sr = cfg.sample_rate
    total = len(audio) / sr

    records = []
    counts = {"written": 0, "no_speech": 0, "multi_speaker": 0}
    for i, s in enumerate(segs):
        a = max(s.start - cfg.window_pad, 0.0)
        b = min(s.end + cfg.window_pad, total)
        rec = {
            "id": f"{uid}_{i:05d}", "source": source, "uid": uid, "series": info.get("series"),
            "episode": info.get("episode"), "idx": i, "cue_start": round(s.start, 3), "cue_end": round(s.end, 3),
            "text": s.text, "speaker_hint": s.speaker_hint, "flags": list(s.flags), "n_cues": s.n_cues,
        }
        if "multi_speaker" in s.flags:
            counts["multi_speaker"] += 1
            records.append(rec)
            continue
        x = audio[int(a * sr):int(b * sr)]
        regions = speech_regions(x, sr, cfg.vad_threshold) if len(x) >= sr // 4 else []
        if not regions:
            rec["flags"].append("no_speech")
            counts["no_speech"] += 1
            records.append(rec)
            continue
        dur = len(x) / sr
        speech = sum(e - s0 for s0, e in regions)
        wav = out_dir / f"{rec['id']}.wav"
        sf.write(wav, x, sr, subtype="PCM_16")
        rec.update(window_start=round(a, 3), window_end=round(b, 3), window_dur=round(dur, 3),
                   speech_ratio=round(min(1.0, speech / dur), 3), wav=str(wav))
        counts["written"] += 1
        records.append(rec)

    tmp = out_dir / "segments.jsonl.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, out_dir / "segments.jsonl")  # atomic: verify workers never see a half-written file
    stats = {"uid": uid, "source": source, "audio_seconds": round(total, 1), **sub_stats, **counts}
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats
