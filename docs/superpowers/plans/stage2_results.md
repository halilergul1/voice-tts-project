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

## Task 3B: narrator pilot (2026-11-02)

`transcribe_narration.py --limit-episodes 20` then `narration_labels.py` on astrobot (tmux `narr`, 3.7 min).

- 20 TRT episodes, 32.8 h audio; gaps between speech cues 11.0 h; Whisper real-time factor 0.006 on the gaps
  (the full 1,017 episodes would take ~3 h).
- 1,162 spans (1.21 h), but only 76 have 6 or more words (13.4 min). Most spans are Whisper's credit
  hallucinations on silence and music ("Bu", "TRT", ".org", "sebeder", "yaptırılmıştır" at random times),
  interjections ("Ah!", "Ne?"), songs ("Yeni yıl ... mutlu olsun", "lay lay"), unsubtitled dialogue and the
  production credits.
- Descriptions with an emotion word: 15, of which about one is a real narrator description ("Süzülen yaş
  yanağına aktı. Zaman yavaş Mustafa sessizce ağlıyor."); the rest are song lyrics. Named 11, resolved to a
  speaker cluster 1. Linked lines: 9 rows; projected to 1,017 episodes: happy ~100 (song lyrics), every other
  class 0. The plan's bar is ~2,000 per class.
- ~~Result: the uploads hold almost no description~~ **Wrong: Whisper skips the narrator.** The listener hears the
  describer in all four check clips (`data/listening_sample/narration_check.html`; clip 5 is the film's own poem
  reading), including two gaps where Whisper wrote nothing. Whisper on the clips alone, with every skip threshold
  off, with a narrator-style prompt, or without timestamps, still jumps over the narration (clip 1: 2.0 s to 16.0 s
  with speech-level audio in between) or writes the credit line "Bu dizinin betimlemesi ..." in its place; it was
  trained on subtitles, which never contain the description. YouTube has one Turkish audio track per episode, so
  the description is in the downloaded audio.
- Two other model families do transcribe it. Clip 2 (gap only): Seamless M4T v2 "Arkasında ki raftan aldığı siyah
  çitli kalın defteri masada açıyor.Farağla birlikte inceliyorlar."; MMS 1b-all "arkasındaki raftan aldığı siyahaç itlii kalın
  depteri masada açıyor ...". Seamless keeps capitals and punctuation (names and sentences need them); on a 14 s
  input it dropped the first half, so it gets chunks of at most 10 s.
- `transcribe_narration.py` now runs Seamless on Silero VAD speech inside the gaps (`narration.speech_chunks`,
  `narration.clean_span_text`). The pilot is re-run with it; the Whisper output is kept as
  `data/dataset/narration_whisper*.jsonl` on astrobot.

### Narrator pilot with Seamless (2026-11-02)

Same 20 episodes, `transcribe_narration.py --batch 32` (Seamless M4T v2 on Silero VAD speech in the gaps, chunks
≤ 10 s), then `narration_labels.py` (bracketed captions such as "(Gülme sesi)", "(Korkma)" dropped; 7 false
descriptions). 4.9 min wall, real-time factor 0.007 on the gaps: **the full 1,017 episodes take ~4 h**.

- VAD speech in the gaps 7.1 h of 11.0 h (narrator plus songs and unsubtitled talk); 5,493 chunks. Real
  descriptions throughout: "Cemha'nın göz bandını çıkardı Ali şaşkınlıkla bakıyor.", "Zehra taksinin arka koltuğunda
  önüne bakarak sessizce ağlıyor.", "Adnan sinirle bıraktı telefonu ...".
- Descriptions with an emotion word: 795 (Whisper: 15). Named 80%; name resolved to a speaker cluster 30% of named
  (Seamless misspells names, and only names with ≥ 5 subtitle speaker hints in one cluster resolve).
- Linked lines (only lines in `data/dataset` can be linked) and projection to 1,017 episodes:

| emotion | descriptions | linked lines | named-and-resolved | no name (first line ≤ 5 s) | e2v top class agrees | projected lines |
|---|---|---|---|---|---|---|
| angry | 110 | 33 | 9 | 24 | 9% | ~1,700 |
| happy | 305 | 117 | 29 | 91 | 34% | ~5,900 |
| sad | 125 | 35 | 8 | 29 | 46% | ~1,800 |
| crying | 37 | 8 | 5 | 3 | 38% | ~400 |
| scared | 101 | 25 | 2 | 24 | 0% | ~1,300 |
| surprised | 117 | 50 | 11 | 39 | 8% | ~2,500 |

- Against the plan's bar (~2,000 linked lines per class): happy and surprised pass, angry and sad fall just short,
  scared and crying well short; none of the new classes reaches the 3,000-clip bar for a style of its own. Most
  links are the weak no-name kind (the next line of any speaker). Only a third of descriptions link to a line;
  linking is CPU-only and can be improved after the GPU pass (fuzzy name match) without transcribing again.
- Stopped per the plan for the user's call on the full run (Step 4).

### Narrator full run (Task 3B Step 4)

2026-11-02, 12:27-16:21 on astrobot (tmux `narr`), the 980 episodes after the pilot; exit 0, no errors in
`data/logs/narration_full.log`. Report in `data/dataset/NARRATION.md`.

- 1,000 episodes, 1,640 h audio; gaps between speech cues 534 h; narrator spans 343 h. Seamless 239 min wall,
  real-time factor 0.007 on the gaps.
- 269,882 spans; 34,207 descriptions with an emotion word; named 80%, name resolved to a speaker cluster 30% of
  named (8,275). 468 lines got conflicting narrator emotions.
- Linked lines (13,858 in all, `narration_styles.jsonl`):

| emotion | descriptions | linked lines | named-and-resolved | no name | e2v top class agrees | e2v ≥ 0.5 |
|---|---|---|---|---|---|---|
| angry | 5,445 | 2,424 | 737 | 1,698 | 12% | 9% |
| happy | 12,867 | 5,528 | 1,668 | 3,955 | 31% | 27% |
| sad | 5,312 | 1,887 | 530 | 1,382 | 33% | 31% |
| crying | 1,366 | 364 | 83 | 281 | 33% | 30% |
| scared | 3,694 | 1,204 | 279 | 931 | 4% | 3% |
| surprised | 5,523 | 2,451 | 726 | 1,734 | 6% | 5% |

- The pilot's projection held (angry came in higher: 2,424 against ~1,700). Against the ~2,000 bar: happy, surprised
  and angry pass, sad just short. No new class reaches 3,000, so crying / scared / surprised cannot become styles of
  their own in stage 2 whatever the ear says.
- Step 5 page: angry, sad, happy, scared, surprised (each ≥ 1,000 linked lines); crying (364) is left out.
  Scared has only ~36 lines with e2v ≥ 0.5, enough for its 20 "agrees" clips.
- Linking stays the weak point (70% no-name links). Fuzzy name matching is CPU-only and can run before Step 5
  without transcribing again.

### Narrator labels by ear (Task 3B Step 5)

2026-11-02, blind page `data/listening_sample/narration_calib.html`: 5 linked lines per class (2 named-and-resolved,
3 no-name, random, one per series), 25 in all at the user's cap instead of the plan's 40 per class. Ratings and key in
`calibration/narration_check_2026-11-02.json`.

| narrator label | heard the same | heard instead |
|---|---|---|
| sad | 4/5 | neutral |
| happy | 2/5 | angry, sad, sad |
| angry | 1/5 | sad, neutral ×2, happy |
| surprised | 1/5 | neutral ×3, other |
| scared | 0/5 | happy, neutral ×2, sad, angry |

- Overall 8/25. Named links 4/10, no-name 4/15. Where emotion2vec+'s top class agreed with the narrator: 4/4; where
  it did not: 4/21.
- The narrator describes the scene, not the voice of the next line: most lines sound neutral or carry a different
  emotion. Scared and surprised are faces and reactions, almost never in the delivery.
- Decision: narrator alone fails the 0.76 bar for angry, happy, scared and surprised, so those labels stay out of
  training, and no new style (crying, scared, surprised) enters stage 2. Sad is 4/5, a pass on very few clips.
  Narrator + emotion2vec+ agreement is right every time, but those clips are ones emotion2vec+ already labels.

Second page at the user's request, label shown (`data/listening_sample/narration_labelled.html`): 8 fresh linked
lines per class (3 named, 5 no-name, none from the first page), answered true / false / unsure. Answers in
`calibration/narration_labelled_2026-11-02.json`.

| narrator label | true (labelled page) | blind page | both |
|---|---|---|---|
| sad | 6/8 | 4/5 | 10/13 (0.77) |
| surprised | 3/8 | 1/5 | 4/13 |
| happy | 2/8 | 2/5 | 4/13 |
| angry | 1/8 | 1/5 | 2/13 |
| scared | 0/8 | 0/5 | 0/13 |

- Seeing the label did not rescue the other classes: 12/40 true overall. Named links 4/15, no-name 8/25.
- Decision: narrator-sad lines (1,887) pass the 0.76 bar on 13 clips and enter training as sad. Angry, happy,
  scared and surprised narrator labels stay out.

### Qwen3-Omni against emotion2vec+ (user's question: is there a stronger, newer model?)

2026-11-02, `scripts/omni_rater.py` on the 300 clips rated by ear on 2026-10-27 (268 with a rating and both
scores). Qwen3-Omni-30B-A3B-Instruct, bf16 split between GPU and CPU (bitsandbytes cannot quantise its fused expert
weights), ~1.7 s per clip; confidence = next-token probability of each class word. Report in
`calibration/OMNI_RATER_qwen3-omni_2026-11-02.md`.

| | Qwen3-Omni | emotion2vec+ |
|---|---|---|
| top class right (angry / happy / sad / neutral) | 175/268 | 172/268 |
| angry, 30 most confident | 25/30 | 30/30 |
| sad, 30 most confident | 20/30 | 23/30 |
| happy, 30 most confident | 11/30 | 16/30 |

- Same overall accuracy, worse at the confident end, which is the part the labels use. emotion2vec+ stays.
  Caveat: the clips were sampled 60 per emotion2vec+ class, which may flatter emotion2vec+ somewhat.

## Task 3C: event clips, same person, no narrator

2026-11-02, `scripts/check_events.py` on astrobot (CPU, ~10 min), `data/dataset_events/event_checks.jsonl`. Narrator =
extension overlaps a Seamless narrator span by ≥ 0.3 s (TRT only); who = subtitle names another speaker cluster;
pitch = extension median F0 ≥ 260 Hz on a male line or ≤ 140 Hz on a female line. All event clips, before the
manifest's evidence gate:

| tag | clips | narrator | who | pitch | flagged | kept |
|---|---|---|---|---|---|---|
| laugh | 5,184 | 1,455 | 69 | 257 | 1,689 | 3,495 |
| breath | 1,222 | 460 | 24 | 78 | 516 | 706 |
| cry | 818 | 317 | 36 | 82 | 383 | 435 |
| cough | 392 | 146 | 16 | 43 | 181 | 211 |
| sigh | 186 | 55 | 1 | 24 | 76 | 110 |
| throat_clear | 145 | 55 | 9 | 13 | 66 | 79 |
| shout | 137 | 79 | 13 | 11 | 88 | 49 |
| moan | 110 | 56 | 4 | 6 | 62 | 48 |
| scream | 66 | 33 | 2 | 4 | 37 | 29 |
| yawn | 54 | 25 | 1 | 3 | 28 | 26 |

- The narrator is the big one: about a third of the clips, since the describer talks in the same gaps as the sounds.
- Cry keeps 435, above the ≥ 300 the plan wants (with unconfirmed). Laugh keeps 3,495.
- Cry check 2 (blind, `data/listening_sample/cry_check2.html`): 30 passing + 10 flagged (5 narrator, 3 pitch,
  2 who), key in `cry_check2.key.json`.

### Cry check 2 by ear (Task 3C Step 4)

User's answers 2026-11-02, `calibration/cry_check2_2026-11-02.json` (narrator wins when narrator words are anywhere
in the clip):

| group | clips | same | other person | narrator | no crying |
|---|---|---|---|---|---|
| pass | 30 | 17 | 7 | 3 | 3 |
| pass, verified (no words in the extension) | 22 | 15 | 6 | 0 | 1 |
| pass, unconfirmed | 8 | 2 | 1 | 3 | 2 |
| flag: narrator | 5 | 0 | 0 | 3 | 2 |
| flag: pitch | 3 | 1 | 2 | 0 | 0 |
| flag: who | 2 | 1 | 0 | 1 | 0 |

- **Gate fails**: 17/30 same (needs ≥ 24), 3 narrator (needs ≤ 1).
- All narrator slips are unconfirmed clips (Whisper heard words in the extension or could not tell). Verified-only
  removes them: 0/22.
- "Another person crying" is the real fault: 6 of 22 verified clips, women crying over a woman's line, which pitch
  cannot see. ECAPA cosine between the line part and the cry part does not separate same from other (both spread
  from -0.07 to 0.31), as with laughs before.
- Flagged clips hold up as rejects: 2 "same" of 10.
- Cry clips that are verified and pass the checks: 306 (after 141, before 132, lead 33), right at `--min-tag` 300.

**Decision (user, 2026-11-03): `[ağlama]` is out of run 2a**, no re-check. A verified-only re-check would likely hear
about 15/22 = 68% same against the 80% bar. Run 2a builds with `--tags-exclude cry`; cry clips hand their place
back to their base clips. Getting `[ağlama]` back is stage 3 work and needs a way to tell whose cry it is (for
example scenes with one person on screen, or a cry detector), not more of the same clips.


## Task 5: training data (2026-11-03)

### Restored-only rebuild (Step 3)

`data/voxcpm_stage2` rebuilt on astrobot (old build kept as `data/voxcpm_stage2_prev`), the Step 3 command with
`--narration-styles data/dataset/narration_styles_sad.jsonl` (sad rows only, 1,912 lines; the narrator's other
classes failed by ear) and `--tags-exclude cry`:

| label | kind | in tier | in slice | written |
|---|---|---|---|---|
| sad | style | 16,254 | 6,002 | yes |
| angry | style | 11,493 | 6,001 | yes |
| happy | style | 22,870 | 6,000 | yes |
| laugh | event | 1,294 | 1,294 | yes |
| breath | event | 252 | 252 | yes |
| cry | event | 128 | 128 | no (excluded) |
| cough | event | 36 | 36 | no (< 150) |

- Train 18,360 / val 315, every train row labelled (34,744 plain corpus rows dropped), no FLEURS rows, 14,753
  labelled rows with a plain same-speaker reference (0.8). Every style at its 6,000 quota; none under 3,000.
- Narrator-sad adds little: of 1,887 linked lines, 1,359 are under 3 s (the tier minimum) and only 307 were restored
  by Sidon (Task 3A ran before the narrator labels existed), so 203 reach the tier and 111 the slice (76 narration
  only, 35 both). Not worth a Sidon pass for run 2a.
- The build now also writes `tier_rows.jsonl` (id, text, duration, split, labelled) for the anchor sentence picker.

### Does Sidon keep the style? (Step 4)

200 written train clips per style, the same clips as `wavs_clean` and `wavs_sidon` (`score_styles.py --wavs`):

| style | hit rate clean | hit rate Sidon | ratio | target post clean / Sidon |
|---|---|---|---|---|
| angry | 0.98 | 0.88 | 0.90 | 0.92 / 0.81 |
| happy | 0.98 | 0.83 | 0.85 | 0.97 / 0.74 |
| sad | 0.92 | 0.78 | 0.85 | 0.78 / 0.62 |

All three above the 0.8× bar, so training stays on `wavs_sidon`. The clean hit rates are near 1 because the labels
were chosen by emotion2vec+ on the raw TV audio (`wavs`), which the separated copy stays close to; Sidon costs 10-15
points of emotion2vec+ hit rate on every style.

### Anchor set (Step 5)

`--limit 200` test (S1 exactly: clip B as prompt and reference, cfg 3.0, no prefix): WER 0.022, UTMOS 3.11 (p10
2.75), voice consistency 0.827, RTF 0.30 (1.67 s per clip of 5.6 s). `anchor_keep` keeps 154 of 200 (77%: 167
with WER 0, 185 with voice cos ≥ 0.75, none read aloud). At 77%, the plan's 1.25 × 18,360 = 22,950 sentences would
give about 17,700 anchors, short of the 18,360 needed, so the sentence file is 25,000 (the first 22,950 unchanged;
synthesis resumes). Full run started 2026-11-03 00:18 in tmux `s2anchor`, marker `data/logs/s2_anchor_done_0`;
estimate about 12 h of synthesis plus 2-3 h of scoring.

Full anchor run done 2026-11-03 ~15:30 (25,000 clips): WER 0.020, CER 0.004, UTMOS 3.12 (p10 2.74), voice consistency
0.830, RTF 0.30. `make_anchor_set.py export` keeps 19,181 of 25,000 (76.7%) in `data/anchor_base` (24 kHz, −20 LUFS).
The sentence file `data/eval/anchor_sentences.txt` stays on astrobot (subtitle text, not committed).

### Final build with anchors (Step 6)

Step 3 command plus `--anchor-dir data/anchor_base --anchor-share 0.5 --anchor-ref data/eval/refs/ref_male_b.wav`
(the no-anchor build is kept as `data/voxcpm_stage2_noanchors`): train 36,720 = 18,360 labelled + 18,360 anchors
(`dataset_id` 2, clip B as ref), val 315; `validate_row` passes; the labelled rows are identical to the no-anchor
build. Non-blind check page `data/listening_sample/stage2_manifest.html` (5 labelled rows, one per written label, +
5 anchors, with their training text).

## Task 6: run 2a (2026-11-03)

LM-only LoRA from the base model, r 32, 4,590 steps (2 epochs of 36,720 rows), `training/voxcpm/stage2_style_lm.yaml`,
started 15:45 in tmux `stage2a` (~0.75 s/step, 24 GB). `scripts/score_stage2.sh` waits in tmux `score2a` and scores
every 400th checkpoint and the last at strength 1.0 once `data/logs/stage2a_train_done` appears: plain Freya 0-39
(S1 setting) against `s0_voxcpm2_contB_freya40`, styles in P2 (neutral + matched text), `[gülme]` and `[nefes]` in
P2. Plain numbers come from the new `scripts/plain_metrics.py` (SQUIM PESQ, UTMOS, Praat CPPS, syl/s; the stage 1
one-off probes in one script, rows in `data/eval/plain_summary.md`). First row, Task 4's untrained plain run on
`style_neutral.txt`: WER 0.0%, PESQ 4.16 (p10 4.01), UTMOS 2.99 (2.60), CPPS 10.84, 6.50 syl/s, consistency 0.860.

Training finished 2026-11-03 16:42 (exit 0); `score_stage2.sh` scored 12 checkpoints at strength 1.0 by 18:40.

### Run 2a by numbers (Task 6 Step 3)

Plain Freya 0-39 in S1 (`data/eval/plain_summary.md`), untrained `s0_voxcpm2_contB_freya40` first:

| run | WER | PESQ (p10) | UTMOS (p10) | CPPS | syl/s | consistency |
|---|---|---|---|---|---|---|
| untrained | 1.1% | 4.24 (4.10) | 3.05 (2.83) | 11.01 | 6.41 | 0.859 |
| step 400 | 0.5% | 4.11 (3.78) | 3.05 (2.38) | 10.51 | 6.63 | 0.865 |
| step 1200 | 0.2% | 4.09 (3.83) | 2.93 (2.58) | 10.14 | 6.77 | 0.856 |
| step 2400 | 0.5% | 4.17 (4.00) | 2.98 (2.67) | 10.28 | 6.50 | 0.872 |
| step 3600 | 0.3% | 4.14 (3.91) | 3.11 (2.72) | 10.22 | 6.59 | 0.866 |
| step 4400 | 0.5% | 4.14 (3.88) | 3.02 (2.72) | 10.11 | 6.59 | 0.868 |
| step 4590 | 0.5% | 4.13 (3.87) | 3.02 (2.62) | 10.18 | 6.25 | 0.865 |

Styles in P2 on neutral text (emotion2vec+ hit rate; gate angry ≥ 0.64, sad ≥ 0.42, happy ≥ 0.42) and tags
(emotion2vec+ is no tag measure; Whisper laugh realised, gate ≥ 0.7):

| run | angry | happy | sad (neutral rate) | laugh realised | voice cos angry / happy / sad |
|---|---|---|---|---|---|
| untrained P2 | 0.40 | 0.15 | 0.05 (0.50) | 0.20 | 0.65 / 0.67 / 0.67 |
| step 400 | 0.10 | 0.20 | 0.00 (0.65) | 0.20 | 0.77 / 0.75 / 0.76 |
| step 1200 | 0.50 | 0.25 | 0.00 (0.60) | 0.30 | 0.73 / 0.73 / 0.70 |
| step 2000 | 0.70 | 0.30 | 0.00 (0.60) | 0.20 | 0.64 / 0.68 / 0.66 |
| step 2400 | 0.75 | 0.15 | 0.00 (0.70) | 0.20 | 0.68 / 0.70 / 0.69 |
| step 3600 | 0.80 | 0.30 | 0.00 (0.70) | 0.30 | 0.67 / 0.71 / 0.69 |
| step 4000 | 0.85 | 0.20 | 0.00 (0.75) | 0.20 | 0.68 / 0.71 / 0.67 |
| step 4400 | **0.90** | 0.15 | 0.00 (0.75) | **0.40** | 0.68 / 0.72 / 0.69 |
| step 4590 | 0.70 | 0.15 | 0.00 (0.65) | 0.30 | 0.67 / 0.71 / 0.68 |

- Plain speech holds: WER below the untrained run everywhere, speed within 0.4 syl/s, voice consistency equal or
  higher, nothing read aloud. The p10 numbers (4th worst of 40 clips) sit 0.1-0.3 below the untrained run (gate
  − 0.1): PESQ p10 3.8-4.0 against 4.10, UTMOS p10 2.4-2.7 against 2.83. Only step 2400 meets PESQ p10; none meets
  UTMOS p10 (best 2.72). A small tail of weaker clips, not a general drop; the blind A/B decides.
- Angry learns steadily (0.40 untrained, 0.90 at step 4400, gate 0.64 from step 2000 on).
- Happy and sad do not move by emotion2vec+: happy 0.15-0.30, sad 0.00 at every checkpoint, sad clips scored
  "neutral" 0.60-0.75. In Task 4 emotion2vec+ disagreed with the ear on synthetic speech both ways, and the user
  heard emotion in the untrained short-prefix clones (2026-11-03, Mac, clip B), so this needs listening before a
  verdict.
- Laugh by Whisper 0.2-0.4 (gate 0.7); tag-set WER 6-12% with laughs (gate 3%, laughter may count as words).
- Styled WER 0-2% on style sets. Voice cos to the plain run 0.67-0.72 (gate 0.70; untrained P2 0.65-0.67).
- **No checkpoint meets every number gate** (sad and happy by emotion2vec+, laugh by Whisper, plain p10). Per Task 6
  the next move is a retry (2b), but the numbers that fail are the ones emotion2vec+ / Whisper got wrong on synthetic
  speech in Task 4, so the candidate is judged by ear first: **step 4400** (best angry and laugh, plain p10 near the
  best).

## Task 7: listening round 1, run 2a step 4400 at strength 1.0 (2026-11-04)

Pages (blind, keys next to them in `data/listening_sample/`), all on Freya lines never used before (200-269, none in
any eval file or the training text):

- `stage2_ab_blind1.html`: 20 plain sentences (Freya 200-219), step 4400 vs untrained, both in S1 (continue + clone
  clip B, cfg 3.0).
- `stage2_style1.html`: 8 clips each of angry / happy / sad (P2: clone clip B + Turkish prefix) and neutral (same,
  no prefix), drawn from Freya 220-239; asks the delivery and "natural?".
- `stage2_tags1.html`: 10 `[gülme]` (Freya 240-249), 10 `[nefes]` (250-259), 10 untagged (260-269), P2; tag at the
  start in 5, at the end in 5.

### Round 1 by ear (Task 7 Step 2, 2026-11-05)

Answers and tallies in `calibration/stage2_round1_2026-11-05.json`.

| page | result | gate | verdict |
|---|---|---|---|
| plain A/B (20) | step 4400 preferred 2, untrained 2, same 16 | untrained ≤ candidate | **pass** |
| style: angry | 0 of 8 (6 heard neutral, 2 happy) | ≥ 6/8 | fail |
| style: happy | 0 of 8 (7 neutral, 1 sad) | ≥ 6/8 | fail |
| style: sad | 2 of 8 (5 neutral, 1 angry) | ≥ 6/8 | fail |
| style: neutral (no prefix) | 6 of 8 | ≥ 6/8 | pass |
| natural yes | angry 6, happy 4, sad 7, neutral 6 (of 8) | ≥ 6/8 | happy fails |
| tag `[gülme]` | 0 of 10 | ≥ 7/10 | fail |
| tag `[nefes]` | 0 of 10 | ≥ 7/10 | fail |
| untagged with a sound | 0 of 10 | ≤ 1/10 | pass |

- **Plain speech holds by ear**: 16 of 20 pairs "same", 2-2 otherwise. The plain p10 dips in the numbers are not
  audible.
- **Styles: run 2a sounds like the untrained model.** The listener hears neutral for 25 of 32 clips; the result
  matches untrained P2 in Task 4 (0 / 0 / 0 of 8). emotion2vec+ scored angry 0.90 on this checkpoint and the ear 0 of
  8, the same gap as P4 angry in Task 4 (1.00 against 0 of 8): the classifier measures something the adapter learned
  from the TV clips that a listener does not hear as anger.
- **Tags: nothing heard as a laugh or a breath.** Listener's note: some clips have "ah ah ah" noises, a bit like a
  breath or a laugh but not natural and not recognisable as either; the adapter makes an attempt, not the sound.
  Untrained P2en (English `[laughter]`) reached 5 of 8 in Task 4.
- The earlier base-model clone demo the user liked (2026-11-03, `tr_clone_short`) used emotional text ("Sana yüz kere
  söyledim, arabama dokunma!"); this page uses neutral Freya text, so the emotion must come from the prefix alone.

**Decision (Task 7 Step 3): styles fail by ear, so back to the Task 6 retry rule: run 2b** (`r: 64, alpha: 64`,
`--label-repeat 2`, same anchors). With 2 repeats the labelled rows are 36,720 and all 19,181 anchors go in (the
0.5 share would want 36,720), so train ≈ 55,900 rows and 2 epochs ≈ 6,990 steps (≈ 1.5 h at 0.75 s/step). A second
failure stops stage 2 for a talk with the user.

## Task 6 retry: run 2b (2026-11-05)

User: "Yes, start 2b". `training/voxcpm/stage2b_style_lm.yaml`: run 2a's recipe with `r: 64, alpha: 64` and the
labelled rows twice. `data/voxcpm_stage2b` = the run 2a manifest command with `--label-repeat 2`: train 55,901
(36,720 labelled + all 19,181 anchors, 34% instead of 50%, since the 0.5 share would want 36,720) / val 315; 6,988
steps (2 epochs). Started 11:50 in tmux `stage2b` (log `~/tts/logs/stage2b_train.log`, marker
`data/logs/stage2b_train_done`).

No number sweep this time: emotion2vec+ and Whisper disagreed with the ear in Task 4 and round 1, so the last
checkpoint goes straight to listening. tmux `s2bpages` (`data/logs/s2b_pages.sh`) waits for the training marker and
makes the round 2 clips at strength 1.0 on Freya lines 270-339 (`data/eval/s2_page2_*.txt`, same layout as round 1;
none in any eval file, the run 2b training text or the anchor sentences): A/B 270-289 (plus the untrained run
`s0_page2_ab`), styles 290-309, `[gülme]` 310-319, `[nefes]` 320-329, untagged 330-339. Marker
`data/logs/s2b_pages_done`.

Training finished 2026-11-05 13:17 (6,988 steps, exit 0; loss/diff 0.82 at the start, about 0.5-0.9 per batch at
the end). Round 2 clips from `step_0006988` at strength 1.0 done 13:22.

## Task 7: listening round 2, run 2b step 6988 at strength 1.0 (2026-11-05)

Same layout as round 1, built on astrobot, keys next to the pages in `data/listening_sample/`:
`stage2_ab_blind2.html` (20 plain pairs, Freya 270-289, vs untrained `s0_page2_ab`, seed 31),
`stage2_style2.html` (8 each angry / happy / sad / neutral from Freya 290-309, P2, "natural?" asked, seed 32),
`stage2_tags2.html` (10 `[gülme]` 310-319, 10 `[nefes]` 320-329, 10 untagged 330-339, P2, seed 33).
