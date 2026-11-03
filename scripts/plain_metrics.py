#!/usr/bin/env python
"""Plain-speech numbers of eval runs, as in the stage 1 tables: WER, read aloud, PESQ (TorchAudio SQUIM estimate,
median and p10), UTMOS (median and p10, from the run's scores.jsonl), CPPS (Praat, dB; lower = hoarser), speaking
rate (Turkish syllables per second of trimmed speech) and voice consistency. Collects the stage 1 one-off probes
(data/compare_stats.py, data/hoarse.py, data/speech_rate.py on astrobot) into one script for stage 2 Task 6.

    ~/tts/venv-voxcpm/bin/python scripts/plain_metrics.py --run s0_voxcpm2_contB_freya40 --run s2a_step400_x1.0_plain

Appends one row per run to data/eval/plain_summary.md. Needs a scored run (eval_tts.py all / score).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from datetime import date
from pathlib import Path

import numpy as np

EVAL_DIR = Path("data/eval")
SUMMARY = EVAL_DIR / "plain_summary.md"
HEADER = ("| run | n | WER | read aloud | PESQ (p10) | UTMOS (p10) | CPPS | syl/s (p10-p90) | voice consistency | date |\n"
          "|---|---|---|---|---|---|---|---|---|---|\n")
VOWELS = re.compile(r"[aeıioöuüâîûAEIİOÖUÜÂÎÛ]")
LABELS = re.compile(r"^\([^)]*\)\s*|\[[^\]]*\]")


def syllables(text: str) -> int:
    return len(VOWELS.findall(LABELS.sub("", text)))


def p10(xs: list[float]) -> float:
    return float(np.percentile(xs, 10))


def run_metrics(run: str, squim, device: str) -> dict:
    import librosa
    import parselmouth
    import torch
    import torchaudio
    from parselmouth.praat import call

    rows = [json.loads(l) for l in (EVAL_DIR / run / "scores.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    pesq, cpps, rate = [], [], []
    for r in rows:
        wav = EVAL_DIR / run / "wavs" / f"{r['k']:04d}.wav"
        y, sr = torchaudio.load(str(wav))
        y16 = torchaudio.functional.resample(y.mean(0, keepdim=True), sr, 16000)
        with torch.no_grad():
            pesq.append(float(squim(y16.to(device))[1].item()))
        s = call(parselmouth.Sound(str(wav)), "Resample", 16000, 50)
        if s.n_channels > 1:
            s = s.convert_to_mono()
        pcg = call(s, "To PowerCepstrogram", 60, 0.002, 5000, 50)
        c = call(pcg, "Get CPPS", "no", 0.01, 0.001, 60, 330, 0.05, "parabolic", 0.001, 0, "Straight", "Robust")
        if c == c:
            cpps.append(c)
        a, _ = librosa.effects.trim(y16[0].numpy(), top_db=35)
        if len(a) > 0.3 * 16000:
            rate.append(syllables(r["text"]) / (len(a) / 16000))
    ut = [r["utmos"] for r in rows if r.get("utmos") is not None]
    cos = [r["spk_cos"] for r in rows if r.get("spk_cos") is not None]
    return {"run": run, "n": len(rows), "wer": st.mean(r["wer"] for r in rows),
            "read_aloud": sum(1 for r in rows if r.get("read_aloud")),
            "pesq": st.median(pesq), "pesq_p10": p10(pesq),
            "utmos": st.median(ut) if ut else None, "utmos_p10": p10(ut) if ut else None,
            "cpps": st.median(cpps) if cpps else None,
            "syl": st.median(rate), "syl_p10": p10(rate), "syl_p90": float(np.percentile(rate, 90)),
            "consistency": st.median(cos) if cos else None}


def fmt(x, d=2):
    return "n/a" if x is None else f"{x:.{d}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from torchaudio.pipelines import SQUIM_OBJECTIVE

    squim = SQUIM_OBJECTIVE.get_model().to(a.device).eval()
    if not SUMMARY.exists():
        SUMMARY.write_text(HEADER, encoding="utf-8")
    for run in a.run:
        m = run_metrics(run, squim, a.device)
        (EVAL_DIR / run / "plain_metrics.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
        row = (f"| {run} | {m['n']} | {m['wer'] * 100:.1f}% | {m['read_aloud']} | {fmt(m['pesq'])} ({fmt(m['pesq_p10'])}) | "
               f"{fmt(m['utmos'])} ({fmt(m['utmos_p10'])}) | {fmt(m['cpps'])} | {fmt(m['syl'])} ({fmt(m['syl_p10'])}-"
               f"{fmt(m['syl_p90'])}) | {fmt(m['consistency'], 3)} | {date.today().isoformat()} |")
        with SUMMARY.open("a", encoding="utf-8") as f:
            f.write(row + "\n")
        print(row)


if __name__ == "__main__":
    main()
