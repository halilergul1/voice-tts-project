# Stage 1 results (2026-10-29, astrobot RTX 5090)

Run: `scripts/train_stage1.sh` with `training/voxcpm/stage1_prosody_lora.yaml` (LoRA r64/a64 on LM + DiT, lr 1e-4,
batch 4 × grad accum 4, 2626 steps = 1 epoch of the 60 h clean-wav slice; 23:50 → 00:19, ~0.67 s/step, peak ~28 GB).
Scoring: `scripts/score_stage1.sh` (voice-design prompt as in stage 0). Val loss/diff 1.059 (step 0) → 1.053 (end):
barely moved.

| checkpoint | corpus WER | corpus CER | corpus UTMOS | Freya100 WER | Freya100 CER | Freya100 UTMOS | UTMOS p10 | voice consistency |
|---|---|---|---|---|---|---|---|---|
| Trendyol-TTS | 0.025 | 0.003 | 2.99 | 0.004 | – | 2.99 | – | 0.763 (495) |
| VoxCPM2 untrained | 0.028 | 0.007 | 2.62 | 0.012 | – | 2.77 | 1.93 (495) | 0.439 (495) |
| step 250 | 0.051 | 0.021 | 2.32 | 0.015 | 0.006 | 2.53 | 1.61 | 0.432 |
| step 500 | 0.049 | 0.021 | 2.37 | 0.011 | 0.005 | 2.51 | 1.60 | 0.434 |
| step 750 | 0.031 | 0.015 | 2.29 | 0.012 | 0.004 | 2.45 | 1.62 | 0.425 |
| step 1000 | 0.043 | 0.013 | 2.33 | 0.017 | 0.008 | 2.49 | 1.70 | 0.413 |
| step 1250 | 0.056 | 0.032 | 2.31 | 0.031 | 0.018 | 2.50 | 1.69 | 0.418 |
| step 1500 | 0.039 | 0.011 | 2.33 | 0.032 | 0.010 | 2.45 | 1.69 | 0.434 |
| step 1750 | 0.046 | 0.013 | 2.34 | 0.018 | 0.008 | 2.45 | 1.78 | 0.429 |
| step 2000 | 0.057 | 0.014 | 2.30 | 0.028 | 0.014 | 2.42 | 1.67 | 0.413 |
| step 2250 | 0.029 | 0.009 | 2.31 | 0.023 | 0.009 | 2.46 | 1.68 | 0.428 |
| step 2500 | 0.037 | 0.012 | 2.36 | 0.025 | 0.010 | 2.42 | 1.64 | 0.422 |
| step 2626 (end) | 0.033 | 0.009 | 2.32 | 0.024 | 0.011 | 2.49 | 1.64 | 0.419 |

Baseline Freya100 numbers are the first 100 sentences of the 495-sentence stage 0 runs.

- Plan rule (lowest corpus CER among checkpoints with Freya WER within 1 point of Trendyol): **step 750**.
- Exit gate on the numbers: **not met.** Freya WER is 0.8 points above Trendyol (gate 0.5), UTMOS p10 is under the
  untrained model at every step, and voice consistency is just under it. The blind A/B (≥ 12/20) is still open.
- UTMOS falls towards the training audio (real TV clips score 2.21), so a lower UTMOS can still sound more natural.
- Failure seen: the voice-design prompt is sometimes read aloud (1 of 50 corpus sentences at step 1250); one Freya
  sentence came out as a single word. Training text never carries a voice description.
- Listening page: `data/listening_sample/stage1_listen.html` (15 sentences; Trendyol, untrained, step 750, step 2626).
