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

## Run 4 (2026-10-30): clean clips only

`make_voxcpm_manifest.py --restored-only` (b675694) keeps only clips Sidon restored (single clips 39,152 of 49,245,
merged 20,424 of 24,470); FLEURS replay restored with `scripts/sidon_fleurs.py` (1,914 of 2,034 readable clips kept:
113 faded endings, 7 changed words). Build `data/voxcpm_v4`: 60.02 h, train 42,050 / val 811, every corpus row,
reference clip and replay row checked as Sidon-restored. Same recipe as run 3, 2629 steps (`checkpoints/stage1_v4`),
30 min training. Command: `--stage 1 --out data/voxcpm_v4 --wavs wavs_sidon --long-wavs wavs_sidon --events-wavs
wavs_clean --max-hours 60 --restored-only --replay-dir data/fleurs_tr_sidon --speaker-gender
data/dataset/speaker_gender.jsonl --voice-desc-lang en`.

| checkpoint | corpus WER | corpus UTMOS (p10) | Freya100 WER | Freya100 UTMOS (p10) | voice consistency |
|---|---|---|---|---|---|
| step 250 | 0.045 | 2.89 (2.34) | 0.022 | 2.98 (2.42) | 0.527 |
| step 500 | 0.040 | 2.75 (2.02) | 0.007 | 2.94 (2.18) | 0.540 |
| step 750 | 0.045 | 2.72 (2.07) | 0.020 | 2.89 (2.29) | 0.529 |
| step 1000 | 0.045 | 3.06 (2.62) | 0.015 | 2.99 (2.41) | 0.534 |
| step 1250 | 0.041 | 2.90 (2.27) | 0.022 | 2.88 (2.33) | 0.541 |
| step 1500 | 0.058 | 2.80 (2.12) | 0.021 | 2.82 (2.17) | 0.541 |
| step 1750 | 0.034 | 2.83 (2.08) | 0.017 | 2.88 (2.25) | 0.538 |
| step 2000 | 0.045 | 2.79 (2.19) | 0.015 | 2.88 (2.20) | 0.521 |
| step 2250 | 0.042 | 2.81 (2.18) | 0.022 | 2.86 (2.14) | 0.530 |
| step 2500 | 0.041 | 2.74 (2.11) | 0.016 | 2.87 (2.27) | 0.537 |
| step 2629 (end) | 0.038 | 2.75 (2.12) | 0.017 | 2.88 (2.33) | 0.528 |

Run 3 end for comparison: corpus WER 0.023, corpus UTMOS 2.65, Freya100 WER 0.019, UTMOS 2.83 (2.24), consistency 0.512.

First 40 Freya sentences, end checkpoints:

| system | PESQ est. (p10) | UTMOS (p10) | WER | female-range | voice consistency |
|---|---|---|---|---|---|
| Trendyol-TTS | 3.89 (3.56) | 3.04 (2.82) | 0.3% | 0% | 0.79 |
| run 3, male prompt | 3.63 (3.06) | 2.99 (2.29) | 1.8% | 32% | 0.52 |
| run 4, male prompt | 3.64 (3.10) | 2.96 (2.39) | 1.3% | 28% | 0.55 |
| run 3, male prompt, cfg 3.0 | 3.65 (3.02) | 3.05 (2.33) | 0.8% | 40% | 0.54 |
| run 4, male prompt, cfg 3.0 | 3.54 (2.87) | 3.11 (2.10) | 0.7% | 30% | 0.59 |
| run 3, clone male clip, cfg 3.0 | 3.51 (3.13) | 2.89 (2.34) | 1.7% | 12% | 0.76 |
| run 4, clone male clip, cfg 3.0 | 3.74 (3.28) | 2.93 (2.65) | 0.5% | 8% | 0.79 |

- With the text prompt run 4 is a small step over run 3 (Freya WER 1.7% vs 1.9%, UTMOS p10 2.33 vs 2.24, voice
  consistency up); the female-range gap stays (Freya100 UTMOS 2.53 female-range vs 3.04 male-range), so the thin voice
  under the text prompt is mostly the prompt picking a random high voice, not only the data.
- Corpus WER rose to 3.8% (run 3 2.3%; 50 TV lines, a handful of words).
- Run 4 + cloning a clean male clip with cfg 3.0 is the best setting so far: WER 0.5%, voice consistency 0.79 and
  female-range 8% (Trendyol 0.3% / 0.79 / 0%), worst-10% PESQ 3.28 and UTMOS 2.65 (run 3 same setting 3.13 / 2.34).
- Listening page: `data/listening_sample/stage1_v4_listen.html`.

### Run 4 best setting on the full scored sets (2026-10-30)

Run 4 end + clone `ref_male.wav` (train clip `ET85vl6YRBE_01406`, 121 Hz) + cfg 3.0, no text prompt
(`scripts/score_stage1.sh` now takes `REF_WAV` and `CFG`):

| system | Freya100 WER | Freya100 UTMOS (p10) | corpus WER | corpus UTMOS (p10) | voice consistency (Freya / corpus) |
|---|---|---|---|---|---|
| Trendyol-TTS | 0.004 | 2.99 | 0.025 | 2.99 (2.55) | 0.76 / 0.78 |
| VoxCPM2 untrained | 0.012 | 2.67 (1.93 on 495) | 0.028 | 2.62 (1.86) | 0.44 / 0.44 |
| run 4, male prompt | 0.017 | 2.88 (2.33) | 0.038 | 2.75 (2.12) | 0.53 / 0.49 |
| run 4 best (clone + cfg 3.0) | 0.013 | 2.86 (2.46) | 0.045 | 2.87 (2.34) | 0.78 / 0.75 |

- Female-range outputs: 9% (Freya), 14% (corpus), against 32% / 46% with the male prompt.
- Corpus WER: most counted errors are spelling variants Trendyol also gets ("valla", "arabada", "rusçarı");
  real slips include a dropped first word in 3 sentences ("Şey,", "Ay", "Garsona") and "harlayayım" heard as "ağırlayayım".
- Exit gate: UTMOS p10 2.46 > 1.93 met; voice consistency 0.78 > 0.44 met; Freya WER 1.3% vs Trendyol 0.4%
  (gate within 0.5 points) not met; listener preference (>= 12 of 20) see below: blind page
  `data/listening_sample/stage1_ab_blind.html` (12 Freya k 40-51 + 8 corpus lines, sides shuffled, seed 20261030).

### Blind A/B result and fast speech (2026-10-30)

Listener result: ours preferred 4, Trendyol preferred 6, same 10. Gate (ours >= 12 of 20) not met, so with the Freya
WER gate also missed, stage 1 run 4 fails the exit gate (2 of 4 met). Listener note: some of ours speak a bit fast.

Speaking rate (syllables per second of speech, Turkish syllables = vowels, leading/trailing silence trimmed;
server `data/speech_rate.py`):

| audio | median syl/s (p10-p90) |
|---|---|
| Trendyol-TTS, Freya 495 | 6.46 (5.70-7.18) |
| VoxCPM2 untrained, Freya 495 | 6.25 (5.16-7.54) |
| run 4, male prompt, cfg 2.0, Freya100 | 6.66 (5.75-8.15) |
| run 4 best (clone + cfg 3.0), Freya100 | 7.58 (6.51-8.68) |
| Trendyol-TTS, corpus 50 | 5.89 |
| run 4 best, corpus 50 | 7.10 |
| run 4 training clips (400 sample) | 5.56 (3.83-7.34) |
| `ref_male.wav` (the cloned clip) | 5.30 |

- The best setting speaks ~17% faster than Trendyol (Freya) and ~20% faster on TV lines; on the 20 blind items the
  median ratio ours/Trendyol is x1.19 and 13 of 20 are more than 10% faster.
- Not inherited: the cloned clip (5.30) and the training clips (5.56) are slower than Trendyol.
- Two causes, same 40 Freya sentences: with the text prompt, guidance speeds speech (run 4 cfg 2.0 6.64 -> cfg 3.0
  7.43; run 3 16 steps 7.03 -> 7.73, 32 steps 7.31 -> 7.48); with cloning, speech is fast at any guidance:

| run 4, clone `ref_male.wav` | syl/s (p10-p90) | PESQ (p10) | UTMOS (p10) | WER | female-range | consistency |
|---|---|---|---|---|---|---|
| cfg 2.0 | 7.39 (6.15-9.19) | 3.56 (2.76) | 2.90 (2.28) | 4.4% | 18% | 0.74 |
| cfg 2.5 | 7.42 (6.42-8.86) | 3.67 (2.94) | 2.96 (2.45) | 1.1% | 12% | 0.78 |
| cfg 3.0 | 7.63 (6.53-8.81) | 3.74 (3.28) | 2.93 (2.65) | 0.5% | 8% | 0.79 |
| Trendyol-TTS | 6.48 (5.83-7.51) | 3.89 (3.56) | 3.04 (2.82) | 0.3% | 0% | 0.79 |

  Lower guidance does not slow cloned speech and costs WER and quality, so cfg 3.0 stays.

Fixes tried on the same 40 sentences (VoxCPM2 `generate` has no speed setting):

| run 4, cfg 3.0 | syl/s | PESQ (p10) | UTMOS (p10) | WER | female-range | consistency |
|---|---|---|---|---|---|---|
| clone `ref_male.wav` (best so far) | 7.63 | 3.74 (3.28) | 2.93 (2.65) | 0.5% | 8% | 0.79 |
| same, slowed with rubberband 0.87 | 6.67 | 1.79 (1.51) | 2.30 (1.89) | 0.8% | 8% | 0.79 |
| same, slowed with atempo 0.87 | 6.66 | 3.72 (3.33) | 2.73 (2.50) | 1.0% | 8% | 0.80 |
| same, slowed with atempo 0.92 | 7.00 | 3.70 (3.28) | 2.80 (2.41) | 0.8% | 8% | 0.80 |
| continue from `ref_male.wav` (`--prompt-wav`) | 7.21 | 3.85 (3.41) | 2.73 (2.40) | 1.1% | 0% | 0.86 |
| continue + clone | 7.18 | 3.82 (3.44) | 2.72 (2.44) | 0.6% | 0% | 0.86 |
| continue, cfg 2.0 | 7.14 | 3.62 (3.30) | 2.80 (2.14) | 2.0% | 0% | 0.84 |

- rubberband (ffmpeg defaults) wrecks the audio; atempo keeps clarity (PESQ) but UTMOS drops ~0.1-0.2, which may be
  the slower pace itself or stretch artifacts: needs listening.
- Continuing from the clip (`eval_tts.py --prompt-wav`, transcript from the clip's .txt) never drifts to a high voice
  (0% female-range) and gives the steadiest voice (0.86, above Trendyol 0.79), is ~6% slower than cloning, but UTMOS
  is lower (2.72).
- Listening page: `data/listening_sample/stage1_speed_fixes.html` (8 sentences where ours is fastest, 5 versions).

### Continue clip on the full scored sets (2026-10-31)

Listener pick from the speed page: "continue clip". Run 4 end, continue from `ref_male.wav` and clone it
(`--prompt-wav` + `--ref-wav`, no text prompt), cfg 3.0; `scripts/score_stage1.sh` takes `PROMPT_WAV`:

| system | WER | PESQ (p10) | UTMOS (p10) | female-range | voice consistency | syl/s |
|---|---|---|---|---|---|---|
| Freya100: Trendyol-TTS | 0.4% | 3.85 (3.60) | 3.01 (2.67) | 0% | 0.79 | 6.38 |
| Freya100: run 4 clone (previous best) | 1.3% | 3.65 (3.00) | 2.87 (2.46) | 9% | 0.78 | 7.58 |
| Freya100: run 4 continue clip | 0.5% | 3.81 (3.40) | 2.74 (2.34) | 0% | 0.86 | 7.18 |
| corpus 50: Trendyol-TTS | 2.5% | 3.83 (3.60) | 3.04 (2.55) | 0% | 0.78 | 5.89 |
| corpus 50: run 4 clone (previous best) | 4.5% | 3.71 (3.15) | 2.90 (2.34) | 14% | 0.75 | 7.10 |
| corpus 50: run 4 continue clip | 2.3% | 3.83 (3.47) | 2.76 (2.38) | 0% | 0.84 | 6.61 |

(WER/PESQ/UTMOS/consistency from `sweep_summary.py` on the same first-N sentences, so Trendyol's Freya figures are
its first 100, not the 495 in `summary.md`.)

- Exit gate: Freya WER 0.5% vs Trendyol 0.4%, within 0.5 points: met (was not met with cloning); UTMOS p10 2.34 >
  1.93 met; consistency 0.86 > 0.44 met. Listener preference pending: round 2 blind page
  `data/listening_sample/stage1_ab_blind2.html` (Freya k 52-63 + corpus k 2, 7, 12, 17, 22, 27, 32, 40, none used in
  round 1; seed 20261031).
- On TV lines it now beats Trendyol on WER, matches its clarity (PESQ 3.83), never drifts to a high voice, and keeps one
  voice better than Trendyol; UTMOS stays ~0.25 lower and speech is still ~12% faster.

### Husky voice and robotic bits: the cloned clip (2026-10-31)

Listener on round 2: speed now fine and ours sounds natural, but Trendyol is cleaner; ours is a bit hoarse/husky and a
tiny robotic sound appears in some words. No preference count given.

Hoarseness measured with Praat (parselmouth, server `data/hoarse.py`; all audio at 16 kHz): CPPS (cepstral peak
prominence, dB; lower = breathier/hoarser, the standard clinical measure) and HNR (harmonics-to-noise, dB; less telling
here). Medians, first 40 Freya sentences:

| audio | CPPS | HNR |
|---|---|---|
| Trendyol-TTS | 10.18 | 11.37 |
| VoxCPM2 untrained, Turkish male prompt | 9.99 | 13.92 |
| run 4, male prompt | 9.18 | 12.37 |
| run 4, clone `ref_male.wav` | 8.73 | 12.60 |
| run 4, continue clip `ref_male.wav` | 8.26 | 11.53 |
| VoxCPM2 untrained, continue clip `ref_male.wav` | 8.34 | 12.41 |
| `ref_male.wav` itself | 7.43 | 12.93 |
| run 4 train clips, Sidon (150) | 8.65 | 13.43 |
| same clips, music removed only | 7.59 | 9.19 |
| same clips, original TV audio | 6.40 | 7.21 |
| FLEURS Sidon / original (100) | 8.79 / 6.93 | 16.75 / 12.89 |

- The husky sound is the cloned clip: `ref_male.wav` is the hoarsest audio measured (CPPS 7.43), and the untrained model
  continuing from it is as husky as run 4 (8.34 vs 8.26).
- Training lowers clearness a little (untrained 9.99 -> run 4 9.18 with the text prompt; the TV training audio is 8.65
  even after Sidon, which already lifts it from 6.40). Across run 4 checkpoints CPPS is flat (8.7-9.9, steps 250-2629),
  so longer training does not make the voice cleaner.
- The corpus has ~47 h of unique Sidon-restored speech (singles 47.12 h; merged clips reuse the same audio) and run 4
  uses all of it, so the plan's retry (`--max-hours 150`) cannot add data; it would only enlarge the adapter.

Clearer male clips (Sidon-kept train clips, 250 random male candidates ranked by CPPS, PESQ >= 3.9; FLEURS male-range
clips by CPPS), then continue + clone, cfg 3.0, run 4 end, 40 Freya sentences:

| setting | CPPS | PESQ (p10) | UTMOS (p10) | WER | female-range | consistency | syl/s |
|---|---|---|---|---|---|---|---|
| Trendyol-TTS | 10.18 | 3.89 (3.56) | 3.04 (2.82) | 0.3% | 0% | 0.79 | 6.48 |
| current clip `ref_male.wav` (CPPS 7.43) | 8.26 | 3.82 (3.44) | 2.70 (2.25) | 0.6% | 0% | 0.86 | 7.18 |
| current clip, 32 steps | 8.34 | 3.85 (3.56) | 2.63 (2.33) | 0.9% | 0% | 0.87 | 7.13 |
| clip A `esrefruya_ep002_00529` (CPPS 10.92, 107 Hz) | 9.91 | 3.74 (3.47) | 3.13 (2.58) | 1.1% | 0% | 0.88 | 7.11 |
| clip B `MSMZa-16Rvg_00230` (CPPS 10.26, 138 Hz) | 10.59 | 4.14 (3.90) | 3.06 (2.83) | 0.2% | 2% | 0.84 | 7.05 |
| FLEURS `4637681251201607381` (CPPS 10.58, 108 Hz) | 10.45 | 4.25 (4.06) | 1.94 (0.85) | 2.1% | 65% | 0.88 | 6.86 |

- Clip B matches or beats Trendyol on every measure but speed: clearness 10.59 vs 10.18, PESQ 4.14 vs 3.89, UTMOS 3.06
  vs 3.04 (p10 2.83 vs 2.82), WER 0.2% vs 0.3%, consistency 0.84 vs 0.79.
- 32 steps changes nothing measurable; the robotic bits are not from too few generation steps.
- The FLEURS clip drifts to a high voice (65%) with very low UTMOS: read-speech clips are no good as the clip.
- Clips copied to `data/eval/refs/ref_male_{a,b,fleurs}.wav` (+ .txt). Listening page:
  `data/listening_sample/stage1_clear_voice.html`.

### Clip B on the full sets; robotic-bit probes (2026-10-31)

Listener: clip B is better (cleaner), but a robotic sound still appears in some words. Clip B, continue + clone, cfg 3.0,
run 4 end (`scripts/score_stage1.sh` best-setting note now points at `ref_male_b.wav`):

| system | WER | PESQ (p10) | UTMOS (p10) | female-range | consistency | CPPS | syl/s |
|---|---|---|---|---|---|---|---|
| Freya100: Trendyol-TTS | 0.4% | 3.85 (3.60) | 3.01 (2.67) | 0% | 0.79 | - | 6.38 |
| Freya100: continue `ref_male.wav` | 0.5% | 3.81 (3.40) | 2.74 (2.34) | 0% | 0.86 | - | 7.18 |
| Freya100: continue clip B | 0.5% | 4.19 (3.93) | 3.08 (2.74) | 6% | 0.83 | 10.49 | 6.97 |
| corpus 50: Trendyol-TTS | 2.5% | 3.83 (3.60) | 3.04 (2.55) | 0% | 0.78 | - | 5.89 |
| corpus 50: continue `ref_male.wav` | 2.3% | 3.83 (3.47) | 2.76 (2.38) | 0% | 0.84 | - | 6.61 |
| corpus 50: continue clip B | 2.8% | 4.16 (3.89) | 3.06 (2.68) | 12% | 0.80 | 10.48 | 6.56 |

- Exit gate with clip B: Freya WER 0.5% vs 0.4% met; UTMOS p10 2.74 > 1.93 met; consistency 0.83 > 0.44 met;
  preference pending: round 3 blind page `data/listening_sample/stage1_ab_blind3.html` (Freya k 64-75 + corpus k 1, 6,
  11, 16, 21, 26, 31, 36, none used in rounds 1-2; seed 20261131).
- Clip B beats Trendyol on PESQ and UTMOS on both sets; female-range creeps back (6% / 12%; clip B is 138 Hz, higher
  than `ref_male.wav` at 121 Hz).

Robotic-bit probes, clip B, 40 Freya sentences (no metric measures the robotic bits; listening page
`data/listening_sample/stage1_robotic_probes.html`, 8 sentences x 6 versions):

| setting | PESQ (p10) | UTMOS (p10) | WER | female-range | consistency | CPPS | syl/s |
|---|---|---|---|---|---|---|---|
| Trendyol-TTS | 3.89 (3.56) | 3.04 (2.82) | 0.3% | 0% | 0.79 | 10.18 | 6.48 |
| clip B, cfg 3.0 | 4.14 (3.90) | 3.06 (2.83) | 0.2% | 2% | 0.84 | 10.59 | 7.05 |
| clip B, cfg 2.5 | 4.13 (3.88) | 3.03 (2.69) | 0.2% | 10% | 0.84 | 10.54 | 7.08 |
| clip B music-removed only (not Sidon) | 3.12 (2.24) | 2.49 (1.95) | 0.9% | 5% | 0.83 | 9.54 | 6.89 |
| clip B, run 4 step 1250 | 4.18 (4.05) | 3.11 (2.81) | 0.5% | 5% | 0.85 | 10.90 | 7.22 |
| clip B, VoxCPM2 untrained | 4.24 (4.10) | 3.07 (2.77) | 1.1% | 8% | 0.85 | 11.01 | 6.41 |
| clip B output + Sidon post-filter (24 kHz) | 4.17 (3.81) | 3.10 (2.64) | 0.5% | 2% | 0.81 | 10.43 | 7.01 |
| clip B seed 1042 | 4.23 (3.97) | 3.06 (2.81) | 0.8% | 8% | 0.84 | 10.42 | 6.99 |
| clip B seed 2042 | 4.15 (3.94) | 3.10 (2.76) | 0.0% | 5% | 0.84 | 10.40 | 7.02 |
| best of 3 seeds (lowest CER, then highest UTMOS) | 4.14 (3.91) | 3.18 (2.91) | 0.0% | 8% | 0.84 | 10.54 | 6.98 |

- The Sidon-restored clip is the right prompt: its music-removed original gives much worse audio (PESQ 3.12).
- Sidon as a post-filter changes nothing measurable (worst-10% PESQ and UTMOS slightly lower).
- Best of 3 takes: WER 0%, UTMOS up, but UTMOS chose the takes so it is inflated; PESQ (not used to pick) is unchanged.
  The 3 seeds won evenly (13 / 13 / 14), so take-to-take variation is real.
- The untrained model with clip B is as clean as run 4 by the metrics and speaks at Trendyol's pace (6.41), but has
  more word errors (1.1%); step 1250 is level with the end checkpoint.

### Round 3 blind result and robotic-bit listening (2026-10-31)

- Blind round 3 (Trendyol vs clip B, 20 new sentences): ours preferred 6, Trendyol 4, same 10. First round ours wins
  head-to-head (round 1: 4 vs 6), but the gate (ours >= 12 of 20) is not met.
- Robotic-probe page, versions the listener liked per sentence (8 sentences): untrained VoxCPM2 in 8 of 8, step 1250 in
  4, clip B end checkpoint in 2, best of 3 in 1, Sidon post-filter in 0. Listener: "the untrained ones are clearly better".
- So the robotic bits come from the stage 1 training, and grow with it (step 1250 < end). Clip B itself is not the
  cause (the untrained model uses the same clip). Likely source: the training audio is Sidon output, a resynthesis from
  w2v-BERT features, and the adapter learns its vocoder character; the metrics do not catch it.
- What training adds: fewer word errors (Freya 0.5% vs untrained 1.1% with clip B). The adapter sits on both the
  language model and the acoustic decoder (DiT), `r 64 / alpha 64` (scaling 1.0), so the plan's retry setting
  (`r 64, alpha 128`) would only double the adapter's strength.

### Adapter strength per part at inference (2026-10-31)

`eval_tts.py --lora-scale LM,DIT` sets each LoRA layer's scaling buffer after loading: the language-model part
(`base_lm` / `residual_lm`, 144 layers) and the acoustic decoder (`feat_decoder`, the DiT, 48 layers); 1 = as trained,
0 = off. Sanity: `1,1` reproduces the default run sample for sample; `0,0` adds exactly zero, but float rounding sends
sampling down another path, so it matches the untrained model only statistically.

Clip B, continue + clone, cfg 3.0, run 4 end, 40 Freya sentences (~400 words, so 0.2% WER is about one word):

| adapter | WER | PESQ (p10) | UTMOS (p10) | female-range | consistency | CPPS | syl/s |
|---|---|---|---|---|---|---|---|
| Trendyol-TTS | 0.3% | 3.89 (3.56) | 3.04 (2.82) | 0% | 0.79 | 10.18 | 6.48 |
| off (untrained) | 1.1% | 4.24 (4.10) | 3.07 (2.77) | 8% | 0.85 | 11.01 | 6.41 |
| full (1,1) | 0.2% | 4.14 (3.90) | 3.06 (2.83) | 2% | 0.84 | 10.59 | 7.05 |
| language only (1,0) | 0.8% | 4.22 (3.99) | 3.12 (2.84) | 5% | 0.84 | 10.71 | 7.37 |
| acoustic only (0,1) | 1.4% | 4.13 (3.98) | 3.10 (2.73) | 10% | 0.84 | 10.54 | 6.27 |
| half (0.5,0.5) | 0.5% | 4.12 (3.90) | 3.12 (2.64) | 10% | 0.85 | 11.00 | 6.74 |

- The word-error gain and the faster pace both come from the language part: language-only 0.8% / 7.37 syl/s,
  acoustic-only 1.4% / 6.27 (untrained 1.1% / 6.41).
- Half strength sits in between on pace (6.74, closest to Trendyol of the trained versions) and is the clearest
  trained version by CPPS (11.00), but its worst-10% UTMOS is lowest (2.64) and high-voice drift is 10%.
- Robotic bits are not measured; listening page `data/listening_sample/stage1_adapter_split.html` (same 8 sentences
  as the robotic-probe page, 5 versions).
