# Stage 0 results (2026-10-28, astrobot RTX 5090)

Environment: `scripts/astrobot_setup_tts.sh` → `~/tts/venv-voxcpm`, torch 2.14.1+cu130, voxcpm 2.0.3, faster-whisper 1.2.1
(needs the cu12 cuBLAS/cuDNN wheels). Runs: `scripts/server_stage0.sh`; cfg 2.0, seed 42, defaults of `eval_tts.py`.

| run | n | WER | CER | UTMOS mean | UTMOS p10 | voice consistency | RTF |
|---|---|---|---|---|---|---|---|
| s0_voxcpm2_zeroshot (Freya) | 495 | 0.018 | 0.005 | 2.67 | 1.93 | 0.439 | 0.29 |
| s0_trendyol (Freya) | 495 | 0.011 | 0.002 | 2.97 | 2.60 | 0.763 | 0.29 |
| s0_voxcpm2_corpus | 50 | 0.028 | 0.007 | 2.62 | 1.86 | 0.436 | 0.29 |
| s0_trendyol_corpus | 50 | 0.025 | 0.003 | 2.99 | 2.55 | 0.777 | 0.29 |
| real corpus clips (data/voxcpm val, wavs_clean) | 50 | – | – | 2.21 | 1.37 | – | – |

Notes
- The plan's UTMOS sanity value (Trendyol ≈ 3.8) has no source: the Trendyol-TTS model card reports no MOS. UTMOSv2 is
  English-trained; our own real TV clips score 2.21, below both systems, so UTMOS ranks systems but is no absolute
  quality score. A LoRA trained on TV audio may lower UTMOS even when it sounds more natural: weigh the blind A/B
  above the UTMOS p10 gate.
- VoxCPM2 voice consistency is low by design here: voice design without a reference voices each sentence anew.
  Compare LoRA runs against it with the same prompt, or give both a fixed reference clip.
- Stage 1 exit-gate targets: Freya WER ≤ 0.016 (Trendyol + 0.5 points), UTMOS p10 ≥ 1.93, voice consistency ≥ 0.439,
  ≥ 12/20 blind A/B preferences over Trendyol.
- Listening page: `data/listening_sample/stage0_baselines.html` (13 sentences, both systems).
