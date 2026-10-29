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

## Run 2 (2026-10-29): fully separated audio + English voice descriptions

Listening to run 1 found two problems: a music/noise bed behind many outputs, and male or female voices regardless of
the prompt. Causes: 80% of training clips had kept their original audio (instrumental at a median -17.8 dB under the
voice, where clean speech measures about -60 dB), and VoxCPM2 ignores Turkish gender prompts (with the untrained model,
"(bir erkek sesi)" gives 83% female-range pitch on 30 Freya sentences, while "(A man's voice)" gives 27% and
"(A woman's voice)" 93%).

Fixes: every training clip vocal-separated (`server_clean_stages.sh` with `SEP_EXTRA="--threshold-db -200
--redo-original" MANIFEST_EXTRA="--max-music-db -40"`: 73.4k separated, 349 original); speaker gender from pitch
(`scripts/speaker_gender.py`: 2,444 speakers, 1,194 female, 615 male, 635 unlabelled); "(A man's voice)" /
"(A woman's voice)" on half of the non-reference rows of labelled speakers (`--speaker-gender ... --voice-desc-lang en`,
8,712 of 42,049 train rows). Same training settings, 2629 steps (`checkpoints/stage1_v2`, 19:08 end). Scored with the
prompt "(A man's voice)" (run 1 used the Turkish prompt).

| checkpoint | corpus WER | corpus CER | corpus UTMOS | Freya100 WER | Freya100 CER | Freya100 UTMOS | UTMOS p10 | voice consistency |
|---|---|---|---|---|---|---|---|---|
| step 250 | 0.053 | 0.008 | 2.41 | 0.028 | 0.011 | 2.45 | 1.71 | 0.451 |
| step 500 | 0.066 | 0.014 | 2.29 | 0.019 | 0.006 | 2.42 | 1.64 | 0.471 |
| step 750 | 0.047 | 0.011 | 2.38 | 0.010 | 0.002 | 2.48 | 1.60 | 0.483 |
| step 1000 | 0.046 | 0.012 | 2.54 | 0.015 | 0.006 | 2.63 | 2.00 | 0.481 |
| step 1250 | 0.060 | 0.013 | 2.40 | 0.011 | 0.003 | 2.46 | 1.69 | 0.485 |
| step 1500 | 0.045 | 0.013 | 2.53 | 0.011 | 0.004 | 2.49 | 1.89 | 0.464 |
| step 1750 | 0.045 | 0.009 | 2.38 | 0.010 | 0.004 | 2.45 | 1.72 | 0.467 |
| step 2000 | 0.043 | 0.008 | 2.44 | 0.010 | 0.004 | 2.45 | 1.72 | 0.472 |
| step 2250 | 0.073 | 0.035 | 2.43 | 0.016 | 0.005 | 2.47 | 1.75 | 0.468 |
| step 2500 | 0.059 | 0.017 | 2.39 | 0.015 | 0.004 | 2.49 | 1.81 | 0.469 |
| step 2629 (end) | 0.042 | 0.008 | 2.40 | 0.007 | 0.002 | 2.48 | 1.82 | 0.469 |

- Plan rule: **step 2629** (corpus CER 0.8%, tied with step 2000; best Freya WER 0.7%).
- Exit gate on the numbers: WER part **met** (0.3 points above Trendyol's 0.4%, gate 0.5); voice consistency 0.47 is
  above the untrained model (0.44) and run 1 (0.42); UTMOS p10 1.82 is still under the untrained 1.93, so that part is
  not met. The blind A/B is still open.
- Background (instrumental level from BS-Roformer on the 100 Freya outputs; clean speech about -60 dB):

  | system | median | p90 | share >= -30 dB |
  |---|---|---|---|
  | Trendyol-TTS | -61.9 | -58.6 | 0% |
  | VoxCPM2 untrained | -62.4 | -29.1 | 11% |
  | run 1 step 750 | -45.7 | -20.2 | 29% |
  | run 2 step 2629 | -62.9 | -50.7 | 3% |

- Gender (30 Freya sentences, median pitch per clip, female range > 165 Hz):

  | system and prompt | median F0 | female-range share | WER |
  |---|---|---|---|
  | untrained, "(A man's voice)" | 144 Hz | 27% | 0.004 |
  | untrained, "(A woman's voice)" | 242 Hz | 93% | 0.019 |
  | run 1 step 750, Turkish male prompt | 193 Hz | 67% | 0.004 |
  | run 2 step 2629, "(A man's voice)" | 137 Hz | 23% | 0.007 |
  | run 2 step 2629, "(A woman's voice)" | 232 Hz | 97% | 0.007 |
  | run 2 step 2629, no prompt | 162 Hz | 50% | 0.019 |

- **Correction:** the corpus columns above are on a different held-out set. Rebuilding the manifest for run 2 rewrote
  `data/eval/corpus_heldout.txt` with 50 new sentences (fixed in e7a50cd; the old set is restored, 2 of its 50
  sentences are in run 2's train text). On the old set (runs `s1v2_step<N>_corpusA`): step 2629 WER 0.059 / CER 0.017
  / UTMOS 2.44, step 2000 0.061 / 0.019 / 2.42, against untrained 0.028 / 0.007 / 2.62 and run 1 step 750
  0.031 / 0.015 / 2.29. Errors are mostly single words ("Gah" dropped, "hallederim" read as "halledelim").
- Listening page: `data/listening_sample/stage1_v2_listen.html` (Trendyol, untrained, run 1, run 2 with male / female /
  no prompt).

### Run 2 listening feedback and diagnosis (2026-10-29)

User: clearly better than run 1, and some sentences sound more natural than Trendyol ("Parktaki yavru köpekler...");
others are a bit robotic or sound like a small room with echo ("Radyoda en sevdiğim şarkı...", "Yemekten sonra
bulaşıkları...").

Reference-free quality estimate (torchaudio SQUIM objective, PESQ estimate 1-4.5; 100 Freya outputs per system, 100
random run 2 train clips):

| audio | PESQ median (p10) | SI-SDR median |
|---|---|---|
| Trendyol-TTS | 3.85 (3.60) | 26.0 dB |
| VoxCPM2 untrained | 3.41 (2.42) | 22.2 dB |
| run 1 step 750 | 2.67 (1.52) | 19.9 dB |
| run 2 step 2629 | 3.09 (1.81) | 19.9 dB |
| run 2 train clips, music removed | 2.58 (1.43) | 16.3 dB |
| the same clips before separation | 1.54 (1.18) | 12.7 dB |

- Fine-tuning moves output quality towards the training audio, which is TV dialogue with room sound, compression and
  separation artefacts. All 73.7k separated clips: 107 h, PESQ median 2.39; >= 2.5: 50.9 h, >= 2.8: 35.1 h,
  >= 3.0: 24.9 h, >= 3.2: 16.0 h. Run 2's train set: >= 3.0 is 13.9 h of 59 h.
- An echo measure from a de-reverb model (anvuew MelBand Roformer: reverb stem vs dry stem) was inconclusive: it finds
  almost no reverb anywhere (-46 to -60 dB) and rates Trendyol highest, so it does not see this kind of room sound.
- Generation settings help a little (30 Freya sentences, run 2 end): guidance 3.0 instead of 2.0 gives PESQ 3.17 vs
  3.03, UTMOS 2.66 / p10 2.15 vs 2.65 / 2.05, WER 0.4% vs 0.7%; 32 diffusion steps PESQ 3.14. Not the main lever.
- Listening check of the estimate: `data/listening_sample/train_quality_bands.html` (train clips in PESQ bands).
