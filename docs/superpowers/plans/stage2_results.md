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

### Short labelled lines: listening check (Task 3D, 2026-11-01)

Blind page, 80 clips of 1.5–3 s: 20 per emotion2vec+ label at the build's cuts, 20 with neutral ≥ 0.9 as a control
(`calibration/short_emotion_check_2026-11-01.json`). The listener heard the label on:

| label | short clips (this check) | longer clips at the same cut (EMOTION_CALIBRATION.md) |
|---|---|---|
| angry | 7 / 20 (35%) | 0.92 |
| happy | 5 / 19 (26%) | 0.79 |
| sad | 9 / 20 (45%) | 0.79 |
| neutral (control) | 16 / 20 (80%) | 0.93 |

Most short "emotional" clips sound neutral (angry 11, happy 11, sad 9 of 20 heard as neutral). The automatic labels
do not hold under 3 s, so `--label-min-dur` is not used for them; emotional lines keep the 3 s minimum. (A human
label, e.g. the narrator's in Task 3B, may still justify short lines later.)

Selection for cleaning with `--max-cer 0.20` only (3 s minimum), before cleaning:

| label | before 3D | with 3D |
|---|---|---|
| happy | 29,050 (47.1 h) | 39,533 (63.5 h) |
| sad | 20,009 (31.7 h) | 25,173 (39.6 h) |
| angry | 13,775 (19.8 h) | 17,868 (25.3 h) |
| laugh | 2,135 | 2,878 |
| breath | 468 | 568 |
| cry | 278 | 344 |

The `lclean` run started at 07:56 (after run 5b's cleaning failed) with the earlier selection that still had the
short lines, so its separation of `data/dataset` (135,426 clips incl. references, ~104k to separate) also covers
them; that costs GPU time only. When run 5b was restarted on `data/dataset` at 13:15, the `lclean` driver was paused
after its `sep_dataset` step (SIGSTOP; its separation workers finish), so the two never run the same step on one
folder. The context cuts now take `--manifests` (only the separated clips the 3 s manifest uses go on to Sidon; the
short lines would have added ~20-30 GB at 96% disk), so the paused driver is not resumed: `data/logs/lclean_restart.sh`
(tmux `lcleanresume`) waits for the `run5b` session to end and for the workers, stops the old driver and starts a
fresh run of the updated script (finished stages are skipped).

## Task 4: base-model probe (2026-11-02)

Untrained VoxCPM2, clip B's voice, cfg 3.0, `scripts/probe_stage2.sh` (39 runs, 00:50 to 01:23 on astrobot). Neutral
text = 20 Freya sentences with no emotion; matched text = 10 val lines per style. Hit rate = emotion2vec+ top class at
the target; voice cos = ECAPA similarity to P0 (plain S1) on the same text; read aloud = label words spoken as words.

| setup | label | hit rate neutral / matched text | neutral rate (neutral text) | read aloud (of 20 / 10) | WER neutral / matched | voice cos | UTMOS |
|---|---|---|---|---|---|---|---|
| P0 plain S1 | - | target neutral 0.50 | 0.50 | 0 | 0.000 | - | 2.99 |
| P0 plain S1 | angry / happy / sad text | 0.10 / 0.30 / 0.00 on matched | - | 0 | 0.000-0.025 | 0.74-0.84 | 3.04-3.18 |
| P1 S1 + Turkish prefix | angry | 0.25 / 0.40 | 0.45 | 7 / 7 | 0.19 / 0.20 | 0.83 | 2.97 |
| | happy | 0.60 / 0.50 | 0.15 | 12 / 6 | 0.15 / 0.21 | 0.86 | 3.01 |
| | sad | 0.00 / 0.00 | 0.40 | 20 / 10 | 0.42 / 0.57 | 0.86 | 3.04 |
| P2 clone + Turkish prefix | angry | 0.40 / 0.40 | 0.25 | 0 / 0 | 0.01 / 0.00 | 0.65 | 2.72 |
| | happy | 0.15 / 0.30 | 0.50 | 0 / 0 | 0.00 / 0.00 | 0.67 | 2.81 |
| | sad | 0.05 / 0.00 | 0.50 | 0 / 0 | 0.01 / 0.00 | 0.67 | 2.87 |
| P3 clone + English prefix | angry | 0.55 / 0.60 | 0.30 | 0 / 0 | 0.01 / 0.00 | 0.66 | 2.40 |
| | happy | 0.60 / 0.40 | 0.10 | 0 / 0 | 0.01 / 0.00 | 0.66 | 2.78 |
| | sad | 0.00 / 0.00 | 0.45 | 0 / 0 | 0.02 / 0.00 | 0.65 | 2.82 |
| P4 styled clip, clone B | angry | 1.00 / 1.00 | 0.00 | 0 / 0 | 0.01 / 0.03 | 0.52 | 2.79 |
| | happy | 1.00 / 0.90 | 0.00 | 0 / 0 | 0.01 / 0.00 | 0.31 | 3.05 |
| | sad | 0.50 / 0.70 | 0.35 | 0 / 0 | 0.01 / 0.06 | 0.41 | 3.19 |

| tag setup | laugh realised (Whisper) | read aloud (of 10): laugh / breath / cry | WER laugh / breath / cry |
|---|---|---|---|
| P1 S1 + Turkish tag | 0.20 | 0 / 0 / 9 | 0.20 / 0.01 / 0.11 |
| P2 clone + Turkish tag | 0.20 | 0 / 0 / 6 | 0.09 / 0.05 / 0.10 |
| P2en clone + English tag | 0.30 | (not measured for English words) | 0.01 / 0.01 / 0.01 |

First reading, before listening:
- **Sad never comes from a text prompt** (0.00-0.05 in P1-P3); only the styled clip (P4) gives it. Angry and happy come
  partly from the English prefix (P3 0.55-0.60) and less from the Turkish one (P2 0.15-0.40).
- **P1 speaks the Turkish prefix**: "(üzgün bir sesle)" read aloud in 20 of 20 sentences, WER up to 0.57. Continuation
  mode treats the prefix as text to say. No style page for P1 (the read-aloud rules it out as an inference recipe).
- **P4 gets the style but not the voice**: voice cos to clip B 0.31-0.52 against ~0.66 for P2 / P3; the model takes
  over the styled clip's speaker. It needs a styled clip of the target voice, which clip B does not have.
- Tags: laughs are rare by Whisper (0.2-0.3); the Turkish cry tag is read aloud ("ağlama", 6-9 of 10).

Listening pages (`data/listening_sample/s2probe/`, keys next to them): `style_P2`, `style_P3`, `style_P4` (8 clips per
style + 8 P0 clips as neutral, "natural?" asked), `tags_P1`, `tags_P2`, `tags_P2en` (8 per tag + 8 P0 clips as none).
Decision (Task 4 Step 4) after the user's answers.

### Probe by ear and decision (Task 4 Step 4, 2026-11-02)

Listener correct of 8 (blind; neutral = 8 P0 clips on the same page; "natural" = yes of answered):

| page | angry | happy | sad | neutral | natural yes (angry / happy / sad / neutral) |
|---|---|---|---|---|---|
| style P2 (clone + Turkish prefix) | 0 | 0 | 0 | 8 | 4/8, 8/8, 7/8, 7/8 |
| style P3 (clone + English prefix) | 0 | 0 | 0 | 7 | 4/8, 7/8, 7/8, 5/8 |
| style P4 (styled clip) | 0 | **8** | 2 | 7 | 7/7, 6/8, 8/8, 5/8 |

| tag page | laugh | breath | cry | none |
|---|---|---|---|---|
| P1 (S1 + Turkish tag) | 0/3 | 1/8 | 0/7 | 4/4 (12 of 32 unanswered) |
| P2 (clone + Turkish tag) | 0/8 | 1/8 | 0/8 | 7/8 |
| P2en (clone + English tag) | 5/8 | 2/8 | 3/8 | 7/8 |

The tag answers were pasted without page names except P2; the first block is taken as P2en, the third as P1 (the
suggested order). The three tag pages share one layout, so only the setup names depend on that.

- **emotion2vec+ disagrees with the ear on synthetic speech.** P4 angry scored 1.00 by emotion2vec+ and 0 of 8 by ear;
  P3 angry / happy 0.55-0.60 and 0 of 8. The style hit rate is a pre-filter at most; Task 6 / 7 decisions need
  listening.
- **English tags make sounds, but not always the speaker's own**: the listener heard a baby crying in the background
  and another man laughing in some P2en clips. Turkish tags make nothing (and "[ağlama]" is read aloud).
- **Decision: every label stays in the training set.** No label meets its gate in a setup that keeps clip B's voice;
  happy reaches 8 of 8 only in P4, which takes over the styled clip's speaker (voice cos 0.31) and needs a happy clip
  of the target voice, which clip B does not have.
- **Styled inference mode for Task 6: P2** (clone clip B + Turkish prefix). P1 reads the prefix aloud (WER up to
  0.57); P2 never does (WER 0.00-0.01), and a training row (prefix + text, reference audio) has the same shape. Both are
  0 of 8 by ear untrained, so the LoRA has to teach the style.
- Open for Task 5 (the user's call): write the sound tags as English words ("[laughter]", "[breath]", "[crying]"),
  which the untrained model already links to the sounds, instead of the Turkish words it ignores or reads aloud.
