# Stage 2 results (astrobot RTX 5090)

Plan: `docs/superpowers/plans/2026-11-01-voxcpm2-stage2.md`.

## Where the emotion labels come from (2026-11-01)

The Engelsiz subtitles describe sounds, not how a line is spoken. Across 988 episodes (190,527 subtitle cues,
`data/dataset/SOUNDTAGS.md` on astrobot):

| cue kind | count | what it is |
|---|---|---|
| music | 89,465 | "müzik - gerilim", "müzik - duygusal", ... |
| sfx | 51,662 | doors, phones, footsteps |
| vocal + background | 40,154 | laugh 28,767, breath 3,631, cry 2,833, cough 1,298, shout 980, ... |
| manner (how a line is said) | 85 | whispering 43, stuttering 18, dialect 12, crying 6, **angry 3**, shouting 2 |

So the subtitles give the sound tags (`[gülme]`, `[nefes]`, `[ağlama]`) but almost no happy / sad / angry labels.
The styles come from emotion2vec+ (automatic, calibrated cuts) and, after Task 3B, from the audio-description
narrator ("... sinirle ...", "... ağlıyor ..."), which a human wrote and which is independent of emotion2vec+.

## Task 3A: labelled clips before cleaning (2026-11-01)

Raw-audio selection build `data/voxcpm_stage2_rawlabels` (labelled clips only, unconfirmed laughs / breaths / cries
included, auto styles at angry 0.70 / sad 0.85 / happy 0.99, no narration labels yet):

| label | text | clips | hours |
|---|---|---|---|
| happy | `(neşeyle)` | 29,050 | 47.1 |
| sad | `(üzgün bir sesle)` | 20,009 | 31.7 |
| angry | `(öfkeyle)` | 13,775 | 19.8 |
| laugh | `[gülme]` | 2,135 | 3.6 |
| breath | `[nefes]` | 468 | 0.7 |
| cry | `[ağlama]` | 278 | 0.5 |

64,664 rows, 101.8 h. Below `--min-tag 150` and not written: cough 114, sigh 73, throat_clear 32, shout 15, moan 15.
Cry is short of the plan's ≥ 300 (with unconfirmed) before any cleaning loss; Task 3C's clean-up will cut it further,
so `[ağlama]` stays the weakest label.

`scripts/server_label_clean.sh` runs in tmux `lclean` on astrobot (log `data/logs/lclean.log`, markers
`data/logs/lclean_<stage>_done`, all done `data/logs/lclean_done`). It waits for the `run5b` chain (more cleaning,
run 5b training and scoring) to end, so the two never write the same folders or share the GPU.

After the run: restored-only counts per label (Task 5 Step 3) go here.

## Task 3D: text recovery (2026-11-01)

Why clips miss the tier: of the emotional single clips, the audio gates (music, trailing silence, sound flags) drop
< 1%; the 3 s minimum drops 45,057 angry / 36,700 happy / 38,107 sad clips, the 5% Whisper-vs-subtitle gate 4,706 /
11,552 / 5,312. Before the dataset, 124,246 lines (64 h) were rejected for a > 20% mismatch (never scored for
emotion). The user listened to 20 clips the text check drops (10 at 5–20%, 10 rejected): the subtitle was mostly
right. Rule: keep the subtitle up to 20%; above, a second recogniser votes (Task 3D in the plan).

Selection with `--max-cer 0.20 --label-min-dur 1.5` (62,078 short labelled lines, 38.1 h, join the tier), before
cleaning:

| label | before 3D | with 3D steps 1–2 |
|---|---|---|
| happy | 29,050 (47.1 h) | 59,235 (76.0 h) |
| sad | 20,009 (31.7 h) | 46,277 (52.4 h) |
| angry | 13,775 (19.8 h) | 38,914 (38.0 h) |
| laugh | 2,135 (3.6 h) | 3,058 (4.9 h) |
| breath | 468 (0.7 h) | 613 (0.9 h) |
| cry | 278 (0.5 h) | 383 (0.6 h) |

147,311 labelled clips, 170.8 h. Cough (161 clips) now passes `--min-tag 150`; it is cleaned but not in scope, so
the stage 2 build passes `--tags-exclude cough`. The `lclean` job uses this selection.

### Second recogniser bench (Task 3D step 3, 2026-11-01)

`scripts/second_asr.py bench`, 300 trusted clips (Whisper = subtitle) and 300 `high_cer` rejects (the span Whisper
aligned, Whisper's text of that span), sharing the GPU with run 5b's cleaning:

| model | trusted CER median / 90th pct | rejects at agree 0.05: subtitle / spoken / drop | at agree 0.15 | clips/s |
|---|---|---|---|---|
| Seamless M4T v2 large | 0.020 / 0.157 | 1% / 6% / 93% | 5% / 15% / 80% | 6.1 |
| MMS 1b-all (tur) | 0.079 / 0.228 | 0% / 3% / 96% | 1% / 9% / 90% | 36.7 |

The rejects are hard audio: both recognisers disagree with each other and with the subtitle on most of them (e.g.
subtitle "Diğer taraflara bakıyorsun", Whisper "diğer taraftan", Seamless "yürürler yürürler"), and the aligned span
often misses words the subtitle has at its edges. At a strict bar the vote rescues ~7% (~4.7k of 67k lines); at
Seamless's own 90th-percentile error (0.15) ~20% (~13k), but then a "spoken" text may carry up to 15% recogniser
errors. Against steps 1–2 (+25k angry, +30k happy, +26k sad clips) this is small and noisy.
