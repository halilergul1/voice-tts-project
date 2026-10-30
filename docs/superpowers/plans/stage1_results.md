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

### Sidon restoration trial (2026-10-29)

User heard band B (SQUIM PESQ 2.0-2.5) and up as good, and asked for a way to clean echo without dropping clips.
Trial of Sidon v0.1 (sarulab-speech/sidon-v0.1, MIT; w2v-BERT 2.0 feature cleaner + vocoder, run as in the official
demo) on 46 run 2 train clips: 14 band A, 13 B, 13 C, 6 E/F as a control. Words scored by Whisper against the
subtitle; voice = ECAPA cosine to the music-removed clip.

| version | WER | UTMOS | PESQ | voice vs music-removed |
|---|---|---|---|---|
| original TV audio | 0.066 | 2.26 | 1.47 | 0.927 |
| music removed (run 2 training audio) | 0.079 | 2.14 | 2.29 | 1 |
| music removed + Sidon | 0.096 | 2.74 | 3.46 | 0.827 (min 0.571) |
| Sidon on original | 0.123 | 3.01 | 3.75 | 0.786 (min 0.470) |

- Sidon lifts quality to about Trendyol's level (Trendyol outputs: UTMOS 2.99, PESQ 3.85), most in bands A-B.
- Cost: about 2 points more word errors after separation (mostly single words: "Arif" heard as "harif", a dropped
  "Bak"), more on raw TV audio (music confuses it), and the voice moves (0.83 median similarity, some clips ~0.57).
- Listening page: `data/listening_sample/sidon_trial.html`.

Listening (user): echo and robotic sound almost gone, many clips much better; but some final words are swallowed
("...Tut, tut parmakları.", "...Unutun ya.", "...siz bilirsiniz."), and one thin female voice stays robotic.
Cause: no timing shift (output lag 0-20 ms), but those clips end with no silence (the subtitle cut lands on the last
word) and Sidon fades the cut-off word: "bilirsiniz" loses its last ~0.3 s. Variants on the same 46 clips:

| variant | WER | PESQ | voice | endings > 6 dB quieter |
|---|---|---|---|---|
| music removed, no Sidon | 0.079 | 2.29 | 1 | – |
| demo padding (1.5 s zeros after) | 0.096 | 3.46 | 0.827 | 13 |
| 0.5 s zeros before | 0.104 | – | – | 15 (by first-pass measure) |
| mirrored end | 0.114 | – | – | 18 (by first-pass measure) |
| 1.5 s of -60 dB noise around | 0.082 | 3.40 | 0.833 | 10 |
| 0.5 s real context from the episode, music removed, noise padding, trimmed back | 0.077 | 3.56 | 0.833 | 8 |

With context Sidon keeps the words (WER equal to the music-removed clips) and gives the best quality. A last-word
level check (last 0.4 s of speech > 6 dB quieter than in the input) flags the remaining 8 for a fallback.
Page: `data/listening_sample/sidon_context.html`.

### Alternatives to Sidon (2026-10-29)

Same 46 clips, same input (0.5 s episode context, music removed, trimmed back, RMS matched). Separate venv
`~/tts/venv-enh` (torch 2.14.1 cu130, `clearvoice`, `resemble-enhance` from GitHub with deepspeed stubbed: inference
does not use it). Miipher-2 (Google) is not public; VoiceBridge has no released code.

| tool | WER | UTMOS | PESQ | voice vs music-removed | endings > 6 dB quieter |
|---|---|---|---|---|---|
| music removed only | 0.079 | 2.14 | 2.29 | 1 | – |
| Sidon v0.1 | 0.077 | 2.76 | 3.56 | 0.833 | 8 |
| ClearerVoice MossFormer2_SE_48K | 0.089 | 2.16 | 2.54 | 0.982 | 2 |
| Resemble Enhance (lambd 0.9, nfe 64) | 0.263 | 2.47 | 3.27 | 0.857 | 1 |
| Resemble Enhance (lambd 0.1) | 0.248 | 2.55 | 3.23 | 0.857 | 1 |

- Resemble Enhance hallucinates Turkish ("Aferin ulan. Vallahi aferin." heard as "kıvırdatın lan vallahi affet
  ediyorum"): unusable here.
- ClearerVoice keeps voice and words but barely changes quality: it does not remove the room sound.
- Sidon stays the pick. Page: `data/listening_sample/restoration_compare.html`.

## Run 3 (2026-10-30): Sidon-restored training audio

`scripts/server_sidon_stages.sh` (a0e7445, ed18db8): every separated clip re-cut with 0.5 s context, music removed,
Sidon, trimmed back; fallback to the music-removed clip on a faded ending (> 6 dB) or changed words (Whisper CER
> 0.05 worse). Kept Sidon: dataset 39,152 of 49,245 (7,092 ending + 3,001 word fallbacks), dataset_long 20,424 of
24,470 (3,407 + 639). Event clips keep wavs_clean (Sidon strips the laughs/breaths they carry). Stage 1 rebuilt:
60.03 h, train 42,013 / val 797, voice descriptions on 8,952 rows. Same recipe, 2626 steps (`checkpoints/stage1_v3`).
Scored with "(A man's voice)"; corpus set is the original one (2 of its 50 sentences in train).

| checkpoint | corpus WER | corpus CER | corpus UTMOS | Freya100 WER | Freya100 CER | Freya100 UTMOS | UTMOS p10 | voice consistency |
|---|---|---|---|---|---|---|---|---|
| step 250 | 0.057 | 0.021 | 2.46 | 0.021 | 0.009 | 2.67 | 1.69 | 0.501 |
| step 500 | 0.050 | 0.022 | 2.71 | 0.026 | 0.009 | 2.84 | 2.13 | 0.491 |
| step 750 | 0.042 | 0.018 | 2.85 | 0.019 | 0.008 | 2.85 | 2.10 | 0.499 |
| step 1000 | 0.065 | 0.023 | 2.60 | 0.021 | 0.011 | 2.75 | 2.19 | 0.484 |
| step 1250 | 0.034 | 0.013 | 2.87 | 0.019 | 0.007 | 2.93 | 2.33 | 0.507 |
| step 1500 | 0.053 | 0.021 | 2.62 | 0.016 | 0.005 | 2.73 | 2.04 | 0.510 |
| step 1750 | 0.050 | 0.020 | 2.68 | 0.012 | 0.004 | 2.75 | 1.97 | 0.523 |
| step 2000 | 0.044 | 0.015 | 2.70 | 0.023 | 0.007 | 2.75 | 2.06 | 0.515 |
| step 2250 | 0.049 | 0.016 | 2.73 | 0.020 | 0.009 | 2.83 | 2.15 | 0.517 |
| step 2500 | 0.052 | 0.018 | 2.66 | 0.017 | 0.008 | 2.79 | 2.07 | 0.511 |
| step 2626 (end) | 0.023 | 0.011 | 2.65 | 0.019 | 0.009 | 2.83 | 2.24 | 0.512 |

Comparison on the same sets (Freya100 = first 100 Freya sentences; corpus = original held-out set):

| system | corpus WER | Freya100 WER | Freya100 UTMOS (p10) | PESQ est. (p10) | music_db p90 | male prompt female-range | female prompt female-range |
|---|---|---|---|---|---|---|---|
| Trendyol-TTS | 0.025 | 0.004 | 2.99 | 3.85 (3.60) | -58.6 | – | – |
| VoxCPM2 untrained | 0.028 | 0.012 | 2.77 (1.93 on 495) | 3.41 (2.42) | -29.1 | 27% | 93% |
| run 2 end | 0.059 | 0.007 | 2.48 (1.82) | 3.09 (1.81) | -50.7 | 23% | 97% |
| run 3 end | 0.023 | 0.019 | 2.83 (2.24) | 3.58 (2.98) | -60.8 | 30% | 80% |

- Sidon training audio lifts output quality above the untrained model on every quality measure (UTMOS, UTMOS p10,
  PESQ estimate) and gives the best TV-line WER of any system (2.3% vs Trendyol 2.5%), background at Trendyol's level.
- Freya WER rose to 1.9% (run 2 0.7%): single-sound slips ("saat üç buçuğa" heard as "sağ üç beşa", "mantı" as
  "mantık"), not dropped chunks. Plan rule (Freya WER within 1 point of Trendyol, then lowest corpus CER) picks
  step 1750 (Freya 1.2%), which once read the prompt aloud ("apropyalı ..."). Gender prompts are followed a little
  less than in run 2.
- Exit gate on the numbers: Freya WER 1.9% (gate 0.9%) not met; UTMOS p10 2.24 > untrained 1.93 met; voice consistency
  0.51 > 0.44 met. Listening page: `data/listening_sample/stage1_v3_listen.html`.

### Run 3: thin, robotic voice in a few outputs (2026-10-30)

Listening feedback: run 3 is clearly better than run 2, but a few outputs still have a very thin, robot-like voice.

Which outputs: per-clip SQUIM PESQ, UTMOS and median F0 on Freya100 and the corpus set. The worst outputs are mostly
female-range voices (median F0 > 165 Hz) produced under the male prompt: female-range UTMOS 2.44 vs male-range 3.04
(Freya100), 2.46 vs 2.98 (corpus). The untrained model's gap is small (2.68 vs 2.82), run 2 has the same gap as run 3.

Generation settings, run 3 end, first 40 Freya sentences (`scripts/eval_tts.py --ref-wav` added for cloning; refs are
two Sidon-kept train clips with PESQ > 4.1, male 121 Hz `ET85vl6YRBE_01406`, female 285 Hz `Sj5vhnLGKHc_00195`):

| setting | PESQ est. (p10) | UTMOS (p10) | WER | female-range | voice consistency |
|---|---|---|---|---|---|
| Trendyol-TTS | 3.89 (3.56) | 3.04 (2.82) | 0.3% | 0% | 0.79 |
| step 1250, default | 3.53 (3.08) | 2.98 (2.31) | 1.8% | 28% | 0.51 |
| end, default (male prompt, cfg 2.0, 16 steps) | 3.63 (3.06) | 2.99 (2.29) | 1.8% | 32% | 0.52 |
| cfg 3.0 | 3.65 (3.02) | 3.05 (2.33) | 0.8% | 40% | 0.54 |
| 32 steps | 3.61 (3.19) | 2.88 (2.42) | 1.9% | 30% | 0.52 |
| 32 steps, cfg 3.0 | 3.51 (2.73) | 3.10 (2.39) | 1.0% | 30% | 0.55 |
| clone male clip (no text prompt) | 3.51 (2.77) | 2.87 (2.43) | 2.9% | 12% | 0.71 |
| clone male clip, cfg 3.0 | 3.51 (3.13) | 2.89 (2.34) | 1.7% | 12% | 0.76 |
| clone female clip | 3.87 (3.42) | 2.78 (2.40) | 1.7% | 100% | 0.72 |

- No setting closes the gap to Trendyol on the worst 10% (UTMOS p10 2.29 to at most 2.43 vs 2.82).
- cfg 3.0 lowers WER (1.8% to 0.8%) but gives more female-range voices under the male prompt.
- Cloning a clean clip pins the voice (consistency 0.52 to 0.76, female-range 32% to 12%), which removes the random
  thin voices the text prompt sometimes picks; cloning the female clip reaches Trendyol's PESQ estimate.
- Longer training: UTMOS is flat from step 1250 to the end (2.98 vs 2.99 here; 2.93 vs 2.83 on Freya100), so more
  steps on the same data is unlikely to fix it.

Training data of run 3 (65.4 h incl. FLEURS replay), SQUIM PESQ on 150 random clips per group:

| group | hours | share | PESQ median (p10) |
|---|---|---|---|
| Sidon kept, female speakers | 20.7 | 32% | 3.50 (2.96) |
| Sidon kept, male speakers | 12.7 | 19% | 3.39 (2.49) |
| Sidon kept, gender unknown | 14.4 | 22% | – |
| fallback (music removed only: faded ending or changed words) | 11.3 | 17% | 1.9 to 2.2 (1.2 to 1.4) |
| FLEURS replay (read speech, not restored) | 6.3 | 10% | 2.49 (1.25) |

- 27% of the training audio is the low-quality kind (fallbacks with separator echo, FLEURS with home-mic recordings),
  the likely source of the thin, robotic voices left in run 3. Female Sidon clips are not worse than male ones.
- 86.4 h of Sidon-kept clips exist in dataset + dataset_long, so a 60 h build can drop every fallback clip.
- Listening page: `data/listening_sample/stage1_v3_settings.html`.
