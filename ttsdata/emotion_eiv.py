"""EmpathicInsight-Voice (LAION, CC-BY-4.0) as an optional second emotion scorer.

The release holds one MLP head per dimension (`model_<Name>_best.pth`) and no inference code; the card's example is
"conceptual". The heads' checkpoints fix the architecture: a projection of the flattened Whisper-small encoder output
(1500 frames x 768 = 1,152,000 -> 64), then Linear 64->64->32->16->1 with two parameter-free layers between the
linears (dropout and an activation; the activation type is not recoverable from the weights, GELU is assumed).
Only the heads behind the coarse mapping (plus valence and arousal) are downloaded (~295 MB each). Scores are intensities on roughly 0-4. Treat the result as experimental: the calibration set is what says whether it
helps on this corpus. Encoder: mkrausio/EmoWhisper-AnS-Small-v0.1 (the card's example), 16 kHz, 30 s max.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

ENCODER_ID = "mkrausio/EmoWhisper-AnS-Small-v0.1"
HEADS_REPO = "laion/Empathic-Insight-Voice-Small"
FRAMES, DIMS = 1500, 768
# fine dimensions that stand for the four coarse classes
COARSE_FROM = {
    "angry": ["Anger", "Contempt", "Impatience and Irritability"],
    "sad": ["Sadness", "Distress", "Disappointment", "Helplessness", "Longing"],
    "happy": ["Amusement", "Elation", "Contentment", "Pleasure/Ecstasy", "Affection", "Triumph", "Relief"],
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z]+", " ", name.lower()).strip()  # "Pleasure/Ecstasy" == "Pleasure_Ecstasy"


def coarse_from_eiv(scores: dict[str, float], min_intensity: float = 1.0, min_margin: float = 0.3) -> str | None:
    """Coarse label from the fine intensities: the strongest coarse group when it is intense enough and clear of the
    runner-up; "neutral" when no group reaches min_intensity; None when two groups are close or nothing was scored."""
    if not scores:
        return None
    got = {_norm(k): float(v) for k, v in scores.items()}
    strength = {c: max((got.get(_norm(d), 0.0) for d in dims), default=0.0) for c, dims in COARSE_FROM.items()}
    top = max(strength, key=strength.get)
    runner = max(v for c, v in strength.items() if c != top)
    if strength[top] < min_intensity:
        return "neutral"
    if strength[top] - runner < min_margin:
        return None
    return top


class EIV:
    def __init__(self, device: str = "cpu"):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        self.device = device if device != "mps" else "cpu"
        self.processor = WhisperProcessor.from_pretrained(ENCODER_ID)
        self.encoder = WhisperForConditionalGeneration.from_pretrained(ENCODER_ID).get_encoder().to(self.device).eval()
        wanted = sorted({d for dims in COARSE_FROM.values() for d in dims} | {"Valence", "Arousal"})
        patterns = sorted({f"model_{v}_best.pth" for d in wanted for v in (d, d.replace("/", "_"), d.replace(" ", "_"),
                                                                           d.replace("/", "_").replace(" ", "_"))})
        heads_dir = Path(snapshot_download(HEADS_REPO, allow_patterns=patterns))  # 15 + 2 heads, ~5 GB, not all 55
        self.heads = {}
        for p in sorted(heads_dir.glob("model_*_best.pth")):
            name = p.name[len("model_"):-len("_best.pth")]
            sd = torch.load(p, map_location="cpu")
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
            self.heads[name] = self._build(sd).to(self.device).eval()

    def _build(self, sd):
        import torch.nn as nn

        head = nn.Module()
        head.proj = nn.Linear(FRAMES * DIMS, 64)
        head.mlp = nn.Sequential(nn.Dropout(0.0), nn.GELU(), nn.Linear(64, 64), nn.Dropout(0.0), nn.GELU(), nn.Linear(64, 32),
                                 nn.Dropout(0.0), nn.GELU(), nn.Linear(32, 16), nn.Dropout(0.0), nn.GELU(), nn.Linear(16, 1))
        head.load_state_dict(sd, strict=True)
        return head

    def embed(self, wav: Path):
        import soundfile as sf
        from math import gcd
        from scipy.signal import resample_poly

        y, sr = sf.read(wav, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != 16000:
            g = gcd(sr, 16000)
            y = resample_poly(y, 16000 // g, sr // g).astype(np.float32)
        y = y[: 30 * 16000]
        feats = self.processor(y, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
        with self.torch.no_grad():
            h = self.encoder(input_features=feats).last_hidden_state[0]  # (1500, 768)
        if h.shape[0] < FRAMES:
            h = self.torch.nn.functional.pad(h, (0, 0, 0, FRAMES - h.shape[0]))
        return h[:FRAMES].reshape(1, -1)

    def score(self, wav: Path) -> dict[str, float]:
        x = self.embed(wav)
        out = {}
        with self.torch.no_grad():
            for name, head in self.heads.items():
                out[name] = round(float(head.mlp(head.proj(x))[0, 0]), 3)
        return out
