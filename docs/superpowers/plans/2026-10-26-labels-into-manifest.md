# Merged clips and subtitle sound labels into the VoxCPM manifest — plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or subagent-driven-development) task by task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make the two new stages pay off in training: merged clips replace their short members in the manifest, subtitle sound labels exclude noisy clips everywhere, and become a Turkish style prefix plus inline non-verbal tags in the Stage 2 (expressive) manifest.

**Inputs (built 2026-10-26, see `ttsdata/clipmerge.py`, `ttsdata/soundtags.py`):**
- `data/dataset/` single clips: `metadata.jsonl`, `quality.jsonl`, `soundtags.jsonl`, `wavs/`, `wavs_clean/`.
- `data/dataset_long/` merged clips: same layout plus `members`, `member_spans`, `member_text_ends`, `gaps`.
- `data/dataset/sound_events.jsonl`: every subtitle sound description with episode time, kind, tag, who, position.
- `soundtags.jsonl` rows: `{"id", "style": [...], "flags": [...], "events": [{"tag", "where": inside|gap:k|before|after, "t", "dt", "char", "in_clip", "who", "who_match", "manner", "desc"}]}`.

**Spec:** `reports/Turkish expressive TTS model options.md` ("Manufacturing emotion, gender and non-verbal labels", "Training stages"); VoxCPM2 style control is a parenthesised instruction at the start of the text (docs, 2026-10-26); Ahenk found Turkish instructions safer than English ones for words starting with "c".

## What the first run showed (local 15 h build, 19 episodes: 9 TRT, 10 Kanal D)

| measure | value |
|---|---|
| merged clips kept (Whisper CER ≤ 0.10) | 1,917 of 2,091 chains, 3.7 h, replacing 4,332 single clips |
| hours at the manifest gate (≥ 3 s, CER ≤ 0.05) | 6.13 h singles only → 7.62 h with merged clips (+24 %) |
| merges joining two speaker clusters | 63 (249 before the stricter cross-cluster / after-question rule) |
| subtitle descriptions mined | 4,066 (music 2,136, sfx 1,033, vocal 372, background laugh 144, overlap 110, censored 81, foreign 31, channel 25, manner 20, unintelligible 16) |
| standalone vocal events per episode | TRT 23.6 (laugh, cry, breath lead), Kanal D 9.5 |
| unique vocal events within 1.5 s of a kept single clip | laugh 87, breath 17, cry 9, shout 4, cough 3, moan 3, sigh 2, scream 1 |
| vocal events inside a clip's audio | 8 (single clips), 1 (merged clips) |
| events in the pause of a merged clip | 0: an event gets its own cue, which never fits a ≤ 0.7 s pause |
| clips with a speaking style from their own line | 22 (crying 7, laughing 4, others 1–2) |
| clips with an exclusion flag (overlap, foreign, unintelligible, censored, channel) | 86 (0.4 %); background laugh track on 43 more |

Scaled to the 1,064-episode TRT catalogue at the TRT rate: about 25,000 vocal event cues (laugh ≈ 13,000, cry ≈ 4,000, breath ≈ 4,000) and about 15,000 of them within 1.5 s of a kept clip. Subtitle speaking styles stay rare (about 1,000 clips catalogue-wide, mostly crying).

Consequences, folded into the decisions below:
- **Non-verbal tags come from Task 5, not from merged pauses.** Re-cutting a clip to include the adjacent event is the only route with volume; decision 5 is reduced to the rare in-audio cases.
- **Emotion styles come from the classifier layer (Task 6).** Subtitle styles are too few to train angry / sad / happy; they become a free check on the classifiers. Task 6 moves before Task 4's Stage 2 manifest.
- **Exclusion flags are cheap and worth it** (Task 2), and cover merged clips too.

## Decisions

1. **One audio, one row.** A merged clip and its members never both enter a manifest. Merged clips win (longer context, real pauses). Members of a merged clip that fails a tier gate stay eligible as singles.
2. **Exclusion flags apply to every stage.** Drop clips flagged `overlap`, `foreign_speech`, `unintelligible`, `censored`, `channel`. Drop `bg_laugh` from Stage 2 only (a laugh track under speech would teach `[gülme]` the wrong sound). `music` is handled by separation, `sfx` stays.
3. **Stage 1 stays plain.** Clips with a strong style (crying, screaming, shouting, laughing, pained) are left out of the Stage 1 prosody manifest instead of being trained with plain text that sounds emotional. They return in Stage 2 with their prefix.
4. **Style prefix, Turkish, VoxCPM syntax.** `"(ağlayarak) Neden gittin?"`. Only styles with at least `--min-style 50` clips in the slice are written; rarer styles leave the clip unprefixed (kept, no label). Unlabelled clips carry no prefix, so "no prefix" keeps meaning "default delivery".
5. **Inline non-verbal tags only where the sound is in the audio and its position is known.** Sounds that label a line (`where: lead`, `(Öksürme sesi) Ben onu mu diyorum?`) come before the words and are usually trimmed off; they join the `before` events in Task 5. A merged clip carries a style only when all members carry it (`style_members` keeps the per-member view). v1: events at `gap:k` of a merged clip, inserted at `member_text_ends[k]`, skipped when `who_match` is false. `inside` events of single clips have no reliable text position: they are not written inline in v1. `before` / `after` events are not in the audio: they wait for Task 5 (event-extended clips).
6. **A tag is trained only at Elise scale.** A tag enters the Stage 2 manifest when the slice holds at least `--min-tag 300` instances; below that it stays metadata. Same for styles at `--min-style`.
7. **Oversample labelled rows** in Stage 2 by `--label-repeat 3` (capped so labelled rows stay under 40 % of the manifest), because emotional speech is a few percent of the corpus.

Vocabulary (Turkish text written into the manifest):

| style | prefix | event | inline tag |
|---|---|---|---|
| crying | (ağlayarak) | laugh | [gülme] |
| shouting | (bağırarak) | cry | [ağlama] |
| whispering | (fısıldayarak) | breath | [nefes] |
| laughing | (gülerek) | sigh | [iç çekme] |
| screaming | (çığlık atarak) | cough | [öksürük] |
| pained | (acı içinde) | scream | [çığlık] |
| scared | (korkarak) | moan | [inleme] |
| angry | (öfkeyle) | yawn | [esneme] |
| happy | (neşeyle) | sniff | [burun çekme] |
| sad | (üzgün bir sesle) | snore | [horlama] |
| surprised | (şaşkınlıkla) | throat_clear | [boğaz temizleme] |
| sarcastic | (alaycı bir tonla) | shout | [bağırış] |
| excited | (heyecanla) | | |
| pleading | (yalvararak) | | |
| breathless | (nefes nefese) | | |
| sighing | (iç çekerek) | | |
| stuttering | (kekeleyerek) | | |

## Review Focus

- **Prefix and tag text must survive the manifest's own gates.** `validate_row` rejects digits only; brackets and parentheses would trip `quality.text_flags` "leftover" if the tier were computed on labelled text. The tier is computed on the unlabelled `normalized_text`; labels are added after. Test: a labelled row passes `validate_row`, and `select_tier` never sees the label.
- **A wrong-speaker event inside a merged pause.** `who_match is False` must suppress the tag; `None` (unknown) is allowed. Test both.
- **Double audio.** No audio file appears twice and no single clip appears next to its merged parent. Test with one merged row and its two members.
- **Rare labels.** A style with 49 clips writes no prefix; one with 50 does. Same for tags at 300.
- **Train/val.** Merged rows never cross splits (the merger already enforces it); the manifest must keep val rows unlabelled-or-labelled the same way as train so eval prompts can test styles.

---

### Task 1: Merged clips in the manifest — DONE 2026-10-26

Implemented as `combine_units` in `ttsdata/manifest.py` and `--long-root` / `--long-wavs` on the manifest script.
Local result (original wavs, no separation on the merged root yet): tier 5.53 h → 6.90 h (+25 %); 1,428 merged clips
in the slice replace 3,251 short clips; references may point at merged clips (own root). The local run used
`--long-wavs wavs`; the server runs `separate_vocals.py --root data/dataset_long` first and uses `wavs_clean`.


**Files:** Modify `scripts/make_voxcpm_manifest.py`; Modify `ttsdata/manifest.py`; Test `tests/test_manifest.py`.

**Interfaces:**
- Produces `combine_units(singles: list[dict], merged: list[dict], quality: dict[str, dict]) -> list[dict]`: rows ready for `tier_funnel`, merged rows that pass the tier replace their members; members of merged rows that fail stay.
- CLI: `--long-root data/dataset_long --long-wavs wavs_clean` (merged audio paths relative to the manifest dir).

- [ ] Test `test_combine_units_prefers_merged_and_never_duplicates`: singles a, b, c; merged m = [a, b] passing the tier → result ids {m, c}; merged failing (cer 0.2) → {a, b, c}.
- [ ] Implement; run `audit_quality.py --root data/dataset_long` and `separate_vocals.py --root data/dataset_long` first (same tools, same layout).
- [ ] Rebuild the local manifest; `tier_report.md` shows hours from merged rows separately. Expected on the 15 h set: the ≥ 3 s pool grows by the merger's reported gain.

### Task 2: Exclusion flags — DONE 2026-10-26

`sound_flags` gate in `tier_funnel` (`EXCLUDE_FLAGS`; stage 2 adds `bg_laugh`), read from `soundtags.jsonl` of both
roots; passes with a warning when `ttsdata soundtags` has not run. Local: 19 clips dropped at the gate.


**Files:** Modify `ttsdata/manifest.py` (`tier_funnel` gains a `flags` gate); Test `tests/test_manifest.py`.

- [ ] Test: a row whose soundtags flags contain `overlap` / `foreign_speech` / `unintelligible` / `censored` / `channel` is dropped at a new `sound_flags` gate; `music`, `sfx` pass; `bg_laugh` passes unless `stage=2`.
- [ ] Implement `sound_flags` gate reading `soundtags.jsonl` from each root (missing file = no flags, gate passes, warning printed).

### Task 3: Label text functions

**Files:** Create `ttsdata/labels.py`; Test `tests/test_labels.py`.

**Interfaces:**
- `STYLE_PREFIX: dict[str, str]`, `EVENT_TAG: dict[str, str]` (tables above).
- `allowed_labels(tags: list[dict], min_style: int, min_tag: int) -> tuple[set[str], set[str]]` (styles and events frequent enough in the slice).
- `labelled_text(text: str, tags: dict | None, styles_ok: set[str], events_ok: set[str]) -> str`.

- [ ] Tests: `labelled_text("Neden gittin?", {"style": ["crying"], "events": []}, {"crying"}, set()) == "(ağlayarak) Neden gittin?"`; two styles → one prefix joined by ", " in table order; a `gap:0` laugh with `char=4`, `who_match=None` → `"Gel. [gülme] Hadi gidelim."`; same with `who_match=False` → unchanged; `inside` event → unchanged; style not in `styles_ok` → unchanged.
- [ ] Implement; prefixes and tags are applied to `normalized_text` after the tier is chosen.

### Task 4: Stage-aware manifest

**Files:** Modify `scripts/make_voxcpm_manifest.py`; Test `tests/test_manifest.py`.

- [ ] CLI `--stage {1,2}`. Stage 1: strong-style rows dropped, no labels written. Stage 2: labels written via `labelled_text`, `--label-repeat 3` with the 40 % cap, `bg_laugh` dropped.
- [ ] `tier_report.md` adds a label table: clips per style / tag in the slice, which were written, which fell under the thresholds.
- [ ] Build both manifests from the local data; read 30 labelled lines by eye; listen to 10 of them.

### Task 5: Event-extended clips (`ttsdata extend-events`) — DONE 2026-10-26

Implemented in `ttsdata/eventclips.py` (+ `ttsdata/labels.py` for the tag vocabulary). Local run, 19 episodes:
219 adjacent events → 135 candidates (84 lost to overlapping neighbours, a 15 s cap or `who_match: false`) → 90 kept
after the Whisper re-check (26 high CER on the re-cut, 17 words in the extension, 2 unaligned): laugh 53, breath 15,
cry 10, throat_clear 3, cough 2, moan 2, shout 2, sigh 2, yawn 1; 0.14 h; 89 of 90 with `who_match: null`.
Scaled to the TRT catalogue (1,064 episodes): roughly 3,000 tagged laughs, 800 breaths, 500 cries, i.e. laugh and
breath clear the 300-instance bar, cry is borderline, the rest stay metadata. Text columns stay plain; the tagged
text lives in `text_tagged` / `normalized_text_tagged` and the csv files.

Open: `verified` is false everywhere until a sound-event detector runs (Task 6); who made the sound is unknown for
nearly every event, so Stage 2 should prefer `before`/`after` events whose detector confirmation also matches the
clip's speaker embedding, or accept the noise knowingly.


The 1.5 s `before` / `after` events are the largest pool of real Turkish laughs, cries and sighs next to their speaker's line, but the clip edges cut them off.

**Files:** Create `ttsdata/eventclips.py`; Test `tests/test_eventclips.py`; CLI subcommand.

**Interfaces:** `extension(clip: dict, event: dict, pad: float = 0.2, max_dur: float = 15.0) -> tuple[float, float] | None` (new span covering the event, or None when it would exceed `max_dur` or the event names another person); output `data/dataset_events/` in the dataset layout with `"[gülme] Hadi!"`-style text built from the plain text plus the tag at the edge.

- [ ] Tests for `extension`: event 0.5 s before → span starts at event start − pad; `who_match False` → None; over `max_dur` → None.
- [ ] Driver reuses `clipmerge._cut_episode` / `_verify` (cut from raw audio, Whisper re-check of the plain text, CER ≤ `merge_max_cer`).
- [ ] A laughter / sound-event detector (layer 2 of the label plan) confirms the extension holds the sound before the row is kept; until it exists, rows are written but marked `"verified": false` and Stage 2 uses only verified ones.

### Task 6a: Sound-event verifier — DONE 2026-10-26 (`ttsdata verify-events`, `ttsdata/eventverify.py`)

Local run on the 90 extended clips, after the user listened to 5 clips per decision: 52 verified (37 laughs: 25 by
Whisper's ha-ha syllables, 3 by the tagger, 9 clean; 9 breaths, 2 cries, 1 sigh, 1 throat clear, 1 yawn clean;
1 cough by the tagger), 29 `speech_in_extension`, 9 `unconfirmed`. The listening found: every `speech_in_extension`
sample really had someone talking in the added part; every verified laugh was real and mostly in the speaker's own
voice (rarely another person); every clean-extension sample had the tagged sound with only the clip's own line
spoken, which is why a clean extension now counts as evidence (`event_accept_clean`). Learned, measured:
- Whisper on the extension alone catches the audio-description narrator (both clips the user flagged, plus more the
  sample missed); the packed whole-clip re-check does not. Speaker similarity (ECAPA) does not separate them.
- PANNs Cnn14 (frame-wise and clip-wise) is not a usable positive detector here: "Music" dominates, user-confirmed
  good laughs score 0.00–0.02, crying and breathing never fire. Whisper's own ha-like syllables are the positive
  evidence for laughs (24 of 53 verified).
- Consequence for Stage 2: `[gülme]` rows are well supported (ha-ha evidence); `[ağlama]` / `[nefes]` rows rest on
  the subtitle cue plus a clean extension (`evidence: clean`), confirmed by listening on a small sample; the laugh's
  speaker is still unverified (rarely another person). The manifest (Task 4) takes `verified` rows by default,
  can restrict to evidence levels, and takes `unconfirmed` rows only on an explicit flag.

### Task 6b: Classifier layer and calibration (from the report)

Out of scope here; listed so the order is clear: emotion2vec+ / EmpathicInsight-Voice coarse emotions on every clip, a laughter / event detector, gender per speaker cluster, 300–500 clips rated by native listeners, agreement measured before any classifier label becomes a prefix. Subtitle labels serve as the free check on the classifiers.

## Order of work

1. ~~Task 2 (exclusion flags) and Task 1 (merged clips in the manifest)~~ done.
2. ~~Task 5 (event-extended clips)~~ done; the manifest side (extended clip replaces its base) lands with Task 4.
3. Task 6 (classifiers + native calibration): the source for angry / sad / happy styles.
4. Task 3 and Task 4 (label text, Stage 2 manifest): once 2 and 3 have produced labels above the thresholds.
