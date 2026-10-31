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
