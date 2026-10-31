"""Aggregate per-sentence TTS scores (scripts/eval_tts.py) into one summary row."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


def lora_config_from_checkpoint(step_dir: Path, allowed: Iterable[str]) -> dict | None:
    """The trainer's lora_config.json of a checkpoint, restricted to the LoRAConfig fields; None when absent.

    VoxCPM's trainer nests the settings: {"base_model": ..., "lora_config": {...}}; a flat file is read as is."""
    f = Path(step_dir) / "lora_config.json"
    if not f.exists():
        return None
    cfg = json.loads(f.read_text(encoding="utf-8"))
    if isinstance(cfg.get("lora_config"), dict):
        cfg = cfg["lora_config"]
    allowed = set(allowed)
    return {k: v for k, v in cfg.items() if k in allowed}


def lora_part(module_name: str) -> str:
    """Which part of VoxCPM2 a LoRA layer sits in: "lm" (base_lm / residual_lm, text and prosody), "dit" (feat_decoder,
    the acoustic decoder that makes the sound) or "other" (projection layers)."""
    if module_name.startswith("feat_decoder"):
        return "dit"
    if module_name.startswith(("base_lm", "residual_lm")):
        return "lm"
    return "other"


def parse_lora_scale(spec: str) -> tuple[float, float]:
    """--lora-scale "LM,DIT" (e.g. "1,0": language-model part as trained, acoustic part off) or one value for both."""
    parts = [float(x) for x in spec.split(",")]
    if len(parts) == 1:
        return parts[0], parts[0]
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError(f"--lora-scale takes LM,DIT or one value, got {spec!r}")


def _mean(scores: list[dict], key: str) -> float | None:
    vals = [s[key] for s in scores if s.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def summarize(scores: list[dict]) -> dict:
    """Means of wer / cer / utmos / spk_cos / rtf over the sentences that have them, plus the UTMOS 10th percentile."""
    utmos = [s["utmos"] for s in scores if s.get("utmos") is not None]
    return {
        "n": len(scores),
        "wer": _mean(scores, "wer"),
        "cer": _mean(scores, "cer"),
        "utmos_mean": float(np.mean(utmos)) if utmos else None,
        "utmos_p10": float(np.percentile(utmos, 10)) if utmos else None,
        "spk_consistency": _mean(scores, "spk_cos"),
        "rtf": _mean(scores, "rtf"),
    }
