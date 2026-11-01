# VoxCPM2 Turkish LoRA, Stage 2 (styles and sound tags) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A model where the user gives any Turkish text plus an emotion (neutral, happy, sad, angry; more if the data allows) and VoxCPM2 speaks that text in that emotion. Built as a language-model-only LoRA that also learns three sound tags (`[gülme]`, `[nefes]`, `[ağlama]`), without making neutral speech any worse than the stage 1 system.

**Product interface (what "done" serves):** `speak(text, emotion)`: `emotion = neutral` → stage 1 setting, no prefix; any other emotion → its Turkish prefix (`(öfkeyle)`, `(neşeyle)`, `(üzgün bir sesle)`, ...) in the styled inference mode Task 4 picks, clip B's voice. The text does not have to fit the emotion ("Saat beşte geliyorum." said angrily), so every style gate is measured on emotion-free sentences first.

**Architecture:** First measure what the base model already does with style words and tags (no training). Then train a fresh LM-only LoRA on a manifest that holds only labelled TV clips plus "anchor" rows: plain sentences spoken by the base model itself with the stage 1 setting, so the adapter has nothing to learn on plain text and stays near the base sound. Every checkpoint is scored for style, tags and plain-sentence quality; a run passes only when a blind round on plain sentences against the untrained model shows it no worse.

**Tech Stack:** Python 3.12, `voxcpm` (`openbmb/VoxCPM2`, in `~/tts/venv-voxcpm` on astrobot), emotion2vec+ large via FunASR (`iic/emotion2vec_plus_large`, `ttsdata.emotion.E2V`), faster-whisper large-v3-turbo, UTMOSv2, ECAPA (`ttsdata.speakers`), pytest.

**Spec:** `reports/Turkish expressive TTS model options.md` ("Manufacturing emotion, gender and non-verbal labels", "Training stages" row 2, "Evaluation protocol" rows Emotion and Non-verbals), overridden where stage 1 taught otherwise: `docs/superpowers/plans/stage1_results.md` (closing summary, rounds 4–6), label calibration in `calibration/emotion_ratings_2026-10-27.csv` and `docs/superpowers/plans/2026-10-26-labels-into-manifest.md` (vocabulary, `--min-tag`, label-first slicing).

## Read this first (human overview)

Stage 1 closed with **no adapter**: untrained VoxCPM2 continuing from and cloning `data/eval/refs/ref_male_b.wav` ("clip B"), cfg 3.0, empty text prompt, beat Trendyol-TTS 18/0/2. Every LoRA trained on our Sidon-restored TV audio pulled the voice towards that audio; the DiT half added robotic bits. So stage 2 trains only what the base cannot do, and protects everything else.

Decisions the user made on 2026-11-01:
1. **Probe the base model first** (Task 4). A style the base already does with some prompt and clip leaves the training set; the LoRA learns only what is missing.
2. **Plain rows come from the base model's own outputs** ("anchor rows", Task 5), not from TV clips. On those rows the base model is already right, so the adapter gets no push away from its sound.
3. **Labels in scope:** happy / sad / angry, `[gülme]`, `[nefes]`, `[ağlama]`. `[ağlama]` has only ~170 clips (below the spec's 300); it is trained and judged by ear, and may fail its gate without failing the stage.
4. **Regression gate:** a blind round of 20 plain sentences against untrained + clip B; the run fails if the listener prefers the untrained model more often than the adapter (ties are fine).
5. **Use what Engelsiz gives us** (same day, after the inventory below): clean every labelled clip, not only the stage 1 slices; take the unconfirmed cries if the listening check passes; mine the audio-description narrator for emotion labels.

### Inventory of Engelsiz labels (astrobot, 2026-11-01)

The build holds 936,260 clips (713 h). The stage 2 manifest so far saw only the 108 h that the stage 1 slices had cleaned, so labels were capped by cleaning, not by data:

| label | in the build (≥ 3 s, calibrated cuts; single + merged clips, overlapping) | cleaned (Sidon) so far |
|---|---|---|
| angry | ~33 h | 6.8 h |
| sad | ~47 h | 9.7 h |
| happy | ~81 h | 11.2 h |
| cry clips (verified + unconfirmed) | 553 | 205 |
| laugh clips | 3,982 | 2,099 |
| breath clips | 850 | 404 |

- Subtitles for the hearing-impaired: 190,527 sound cues; vocal ones laugh 28,767, breath 3,631, cry 2,833, cough 1,298, shout 980, moan 562, scream 475, sigh 404. Speaking-manner cues ("(ağlayarak)", "(fısıldayarak)") are only 85: the subtitles describe sounds, not emotions.
- Cry funnel: 2,833 cues → 1,156 next to a usable line (2,476 are standalone) → 818 after the Whisper check → 362 verified + 191 unconfirmed (sound without words; no detector can confirm a cry) + 265 with speech in the extension (excluded) → 152 in the current tier.
- Audio description for blind viewers (TRT, "Sesli Betimleme Derneği"): a narrator speaks between dialogue lines and names characters' emotions ("Ayşe'nin saçlarını okşarken ağlıyor", "Sade Nur sevinçle gülümsedi", "Şivenaz zindanda sinirle dört dönerken"). 40 of 952 sampled narrator snippets carry an emotion word. Never transcribed so far: the pipeline only cut audio where subtitle lines are. Subtitle lines carry a character name on 73,726 clips (8 %), which links a description to a speaker.

Order of operations:
1. **Scoring code** (Task 1, Mac): label-aware WER, style hit rate with emotion2vec+, laugh detection from Whisper, `--lora-scale` for LM-only adapters.
2. **Blind page builder** (Task 2, Mac): the A/B page and a new "which style is this?" page, as repo code.
3. **Manifest options** (Task 3, Mac): labelled rows only, neutral references for labelled rows, anchor rows from a folder, unconfirmed event clips per tag.
3A. **Label-first cleaning** (Task 3A, GPU ~10 h): separate + Sidon every emotion-labelled clip, separate every usable event clip.
3B. **Narrator emotion labels** (Task 3B, GPU pilot then ~3–5 h): transcribe the narrator, link "X sinirle..." to X's lines, calibrate by ear.
3D. **Text recovery** (Task 3D, Mac + GPU): keep the subtitle up to a 20% mismatch, labelled lines from 1.5 s, and a second recogniser to rescue lines rejected for a bigger mismatch. Done before any stage 2 training.
3C. **Event clip clean-up** (Task 3C, CPU): drop event clips where the narrator talks over the sound or the sound belongs to another person than the speaker; second cry check by ear.
4. **Base probe** (Task 4, GPU ~1 h): styles and tags with the untrained model in four prompt / clip setups. Decision: which labels the LoRA trains, and which inference setup styled sentences use.
5. **Data** (Task 5, GPU): check Sidon keeps the styles; synthesise the anchor set; rebuild `data/voxcpm_stage2`.
6. **Train and score** (Task 6, GPU): run 2a, LM-only, fresh from base; score checkpoints; sweep `--lora-scale`.
7. **Gates** (Task 7): style page, tag page, plain-sentence regression round; results in `docs/superpowers/plans/stage2_results.md`.

Tasks 1–3 are code with unit tests on the Mac; Task 3B's parsing code too. Tasks 3A, 3B (transcription) and 4–7 run on astrobot and **each GPU run needs the user's go-ahead first.**

## Global Constraints

- Server: `ssh ibrahim@astrobot.local`; unreachable while Cisco VPN or Cloudflare WARP is on (ask the user to switch it off). Project dir `~/voice-data`; code goes there with `scp` (or `rsync` without `--delete`) from this repo. TTS venv `~/tts/venv-voxcpm`; do not install into `~/voice-data/.venv`.
- Before any GPU job: `tmux ls` and `nvidia-smi` on astrobot. `run5b` (`scripts/server_more_clean.sh`: cleaning, then a stage 1 run 5b LoRA) may still be running; ask the user whether to wait or stop its training part. Never stop a cleaning step.
- Ask the user before every GPU run (probe, anchor synthesis, training, scoring). Long jobs run detached (tmux or `Popen(start_new_session=True)`) with a done marker in `data/logs/`.
- Stage 1 inference setting ("S1"): `eval_tts.py synth --system voxcpm2 --model-dir ~/tts/models/VoxCPM2 --prompt '' --prompt-wav data/eval/refs/ref_male_b.wav --ref-wav data/eval/refs/ref_male_b.wav --cfg 3.0` (16 steps, seed 42). It is the baseline for every comparison.
- LoRA: `enable_lm: true`, `enable_dit: false`, `enable_proj: false`, trained **fresh from `~/tts/models/VoxCPM2`** (no stage 1 adapter). `training_cfg_rate` left at default. Inference LoRA settings must match training (`eval_tts.py` reads `lora_config.json`).
- Label text, exactly as `ttsdata/labels.py`: styles `(neşeyle)`, `(üzgün bir sesle)`, `(öfkeyle)` as a prefix; tags `[gülme]`, `[nefes]`, `[ağlama]` inline at the edge where the sound is. No other spellings in training text. Other tags (`[iç çekme]` 73 clips, `[öksürük]` 114) stay metadata.
- Emotion labels: emotion2vec+ at the calibrated cuts `angry=0.70,sad=0.85,happy=0.99` (`--auto-styles --auto-conf ...`). Event clips: evidence `laugh_syllables,tagger,clean`; `--min-tag 150` (so `[ağlama]` is written).
- Audio: speech clips Sidon-restored only (`--wavs wavs_sidon --long-wavs wavs_sidon --restored-only`); event clips keep `--events-wavs wavs_clean` (Sidon removes laughs and cries). Replay: no FLEURS in stage 2 (anchors replace it).
- Sentence sets: Freya-TR-Eval lines 0–119 were used in stage 1 rounds; stage 2 uses lines 120–199 for style / tag probes and 200+ for blind regression rounds, 20 new lines per round, never reused.
- Commit only files this plan creates or changes. Never stage `README.md` or `notebooks/`. Never stash, reset or switch branches. Push to `main`.

## Review Focus

- **Label words spoken aloud.** A model that reads "gülme" or "öfkeyle" as a word is the most likely visible failure. Test: `read_aloud("öfkeyle neden geldin", "(öfkeyle) Neden geldin?") == ["öfkeyle"]`, and `read_aloud` is reported per run and gated at 0.
- **WER punished for a laugh.** Whisper writes laughter as "hahaha"; the reference keeps "gülme" after `for_cer`. Both would count as errors and hide real word errors. Test: `scoring_texts("[gülme] Hadi gidelim.", "Hahaha hadi gidelim")` returns equal strings.
- **Continuation overrides the style.** In continue mode the model copies the prompt clip's delivery (clip B is neutral). Task 4 measures style hit rate in both modes; Task 6 scores styled sentences in the mode Task 4 picks, never silently in S1.
- **The reference teaches the style instead of the text.** If a labelled row's `ref_audio` is another angry clip, the model can learn "angry because the ref is angry" and ignore `(öfkeyle)`. Test: `pair_neutral_refs` never returns a labelled clip and never the target itself.
- **`--lora-scale` on an LM-only adapter.** `eval_tts.py` exits when a part has 0 LoRA layers, so `--lora-scale 0.5,0` on a run 2a checkpoint would fail. Test: `check_scale_parts({"lm": 144, "dit": 0}, 0.5, 0.0)` passes; `check_scale_parts({"lm": 144, "dit": 0}, 0.5, 1.0)` passes (scale 1 on a missing part changes nothing); `check_scale_parts({"lm": 0, "dit": 48}, 0.5, 0.0)` raises.

---

### Task 1: Label-aware scoring (`ttsdata/stylemetrics.py`, `scripts/score_styles.py`, `scripts/eval_tts.py`)

**Files:**
- Create: `ttsdata/stylemetrics.py`
- Create: `scripts/score_styles.py`
- Modify: `scripts/eval_tts.py` (`cmd_score`: label-aware texts and a `read_aloud` field; `cmd_synth`: `check_scale_parts` instead of the inline `if not n["lm"] or not n["dit"]` exit)
- Modify: `ttsdata/evalmetrics.py` (add `check_scale_parts`)
- Test: `tests/test_stylemetrics.py`, `tests/test_evalmetrics.py`

**Interfaces:**
- Consumes: `ttsdata.labels.STYLE_PREFIX`, `EVENT_TAG`; `ttsdata.textnorm.for_cer`; `ttsdata.eventverify.laugh_syllables`; `ttsdata.emotion.E2V`, `COARSE`; `ttsdata.speakers.load_encoder`, `embed_clips`.
- Produces:
  - `strip_labels(text: str) -> str`: drops a leading `(...)` prefix and every `[...]` tag, squeezes spaces.
  - `scoring_texts(reference: str, hypothesis: str) -> tuple[str, str]`: both through `for_cer`, reference after `strip_labels`; when the reference had a tag, hypothesis tokens with `laugh_syllables(tok) >= 2` are dropped.
  - `read_aloud(hypothesis: str, reference: str, extra_words: tuple[str, ...] = ()) -> list[str]`: label words (inner text of every `STYLE_PREFIX` / `EVENT_TAG` value, plus `extra_words` such as the English probe prompts) present in `for_cer(hypothesis)` but not in `for_cer(strip_labels(reference))`.
  - `style_hit(post: dict[str, float], target: str) -> bool`: argmax over `COARSE` equals target.
  - `style_summary(rows: list[dict], target: str) -> dict` with keys `n, hit_rate, target_post, neutral_rate` (rows carry `e2v` posteriors).
  - `tag_realised(asr: str, tag: str) -> bool | None`: `laugh` → `laugh_syllables(asr) >= 2`; other tags → `None` (ears only).
  - `check_scale_parts(counts: dict[str, int], lm: float, dit: float) -> None` in `evalmetrics.py`: raises `ValueError` when a part whose scale is not 1 has 0 layers.
  - `scripts/score_styles.py --run NAME [--wavs DIR] --target {angry,happy,sad,neutral} [--tag laugh|breath|cry|sigh] [--voice-ref-run NAME]` → `data/eval/<run>/styles.jsonl` rows `{k, e2v, hit, tag_realised, ref_cos}` and one row appended to `data/eval/style_summary.md` (`| run | target | n | hit rate | target post | neutral rate | tag realised | read aloud | voice cos | date |`). `--wavs` scores any folder of wavs (used for the Sidon check in Task 5). `ref_cos` = ECAPA cosine to the mean embedding of `--voice-ref-run` (the S1 plain run), so a style that changes the voice shows up.
  - `scores.jsonl` rows gain `read_aloud: list[str]`; `summary.md` rows unchanged.

- [ ] **Step 1: Write the failing tests**

```python
from ttsdata.stylemetrics import strip_labels, scoring_texts, read_aloud, style_hit, style_summary, tag_realised
from ttsdata.evalmetrics import check_scale_parts
import pytest

def test_strip_labels():
    assert strip_labels("(öfkeyle) Neden geldin?") == "Neden geldin?"
    assert strip_labels("[gülme] Hadi gidelim. [nefes]") == "Hadi gidelim."
    assert strip_labels("Saat beşte (belki) gelir.") == "Saat beşte (belki) gelir."  # only a leading prefix

def test_scoring_texts_ignore_laughter_only_when_tagged():
    assert scoring_texts("[gülme] Hadi gidelim.", "Hahaha hadi gidelim") == ("hadi gidelim", "hadi gidelim")
    assert scoring_texts("Hadi gidelim.", "Hahaha hadi gidelim") == ("hadi gidelim", "hahaha hadi gidelim")

def test_read_aloud():
    assert read_aloud("öfkeyle neden geldin", "(öfkeyle) Neden geldin?") == ["öfkeyle"]
    assert read_aloud("gülme hadi", "[gülme] Hadi.") == ["gülme"]
    assert read_aloud("gülme sakın", "Gülme sakın.") == []          # the word is in the sentence itself
    assert read_aloud("angry neden", "Neden?", extra_words=("angry",)) == ["angry"]

def test_style_hit_and_summary():
    p_ang = {"angry": 0.6, "neutral": 0.3, "sad": 0.1, "happy": 0.0, "fearful": 0.9}  # non-coarse classes ignored
    assert style_hit(p_ang, "angry") and not style_hit(p_ang, "sad")
    rows = [{"e2v": p_ang}, {"e2v": {"angry": 0.1, "neutral": 0.8, "sad": 0.1, "happy": 0.0}}]
    s = style_summary(rows, "angry")
    assert s["n"] == 2 and s["hit_rate"] == 0.5 and s["neutral_rate"] == 0.5 and abs(s["target_post"] - 0.35) < 1e-9

def test_tag_realised():
    assert tag_realised("Hahaha hadi gidelim", "laugh") is True
    assert tag_realised("hadi gidelim", "laugh") is False
    assert tag_realised("hadi gidelim", "cry") is None

def test_check_scale_parts_lm_only_adapter():
    check_scale_parts({"lm": 144, "dit": 0}, 0.5, 0.0)
    check_scale_parts({"lm": 144, "dit": 0}, 0.5, 1.0)
    with pytest.raises(ValueError):
        check_scale_parts({"lm": 0, "dit": 48}, 0.5, 0.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_stylemetrics.py tests/test_evalmetrics.py -v`
Expected: FAIL with `ModuleNotFoundError: ttsdata.stylemetrics` (and `ImportError` for `check_scale_parts`).

- [ ] **Step 3: Implement `ttsdata/stylemetrics.py` and `check_scale_parts`**

Pure functions only; `E2V` and ECAPA are imported inside `scripts/score_styles.py`, not here. `read_aloud` matches multi-word labels ("üzgün bir sesle") as a substring of the normalised hypothesis.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all PASS (the existing suite included).

- [ ] **Step 5: Wire into `scripts/eval_tts.py` and write `scripts/score_styles.py`**

`cmd_score`: `ref, hyp = scoring_texts(sentences[k], asr_text)`; WER/CER on those; add `read_aloud(asr_text, sentences[k], extra_words=<inner words of run.json "prompt">)`. The text prompt itself (`--prompt`) is never part of the reference. `cmd_synth`: replace the exit with `check_scale_parts(n, *lora_scale)` (message unchanged on failure). `score_styles.py` runs in `~/tts/venv-voxcpm` (FunASR is in the server venv since 2026-10-27; install the `emotions` extra there only if the import fails, and say so in the results).

- [ ] **Step 6: Smoke test on the Mac** (3 wavs, CPU)

Run: `.venv/bin/python scripts/score_styles.py --wavs data/eval/smoke_voxcpm2/wavs --target neutral --run smoke_style`
Expected: `data/eval/smoke_style/styles.jsonl` with one row per wav and a row in `data/eval/style_summary.md`.

- [ ] **Step 7: Commit**

```bash
git add ttsdata/stylemetrics.py ttsdata/evalmetrics.py scripts/score_styles.py scripts/eval_tts.py tests/test_stylemetrics.py tests/test_evalmetrics.py
git commit -m "Stage 2 scoring: label-aware WER, style hit rate, laugh realisation, LM-only --lora-scale"
```

---

### Task 2: Blind pages as repo code (`ttsdata/blindpage.py`, `scripts/blind_page.py`)

Stage 1 page builders live only on astrobot (`data/ab_page6.py`). Stage 2 needs the A/B page again and a new style page, so they move into the repo.

**Files:**
- Create: `ttsdata/blindpage.py` (pure: keys, shuffling, tallies)
- Create: `scripts/blind_page.py` (HTML writer; start from `scp astrobot:~/voice-data/data/ab_page6.py` and keep its look: inline audio as data URIs, CSS tokens with dark mode, radio picks kept in `localStorage`, a "Show result" button)
- Test: `tests/test_blindpage.py`

**Interfaces:**
- Produces:
  - `ab_key(n: int, seed: int) -> list[str]`: per pair, the side ("A"/"B") our system is on.
  - `tally_ab(key: list[str], picks: list[str]) -> dict` with `ours, theirs, same` (picks are "A", "B" or "same").
  - `style_items(by_style: dict[str, list[str]], n_per: int, seed: int) -> list[dict]`: `{"wav", "style"}` items, `n_per` from each style, shuffled; the same sentence index never appears twice in a row.
  - `tally_style(items: list[dict], picks: list[str]) -> dict` with `per_style: {style: {"correct", "n"}}` and `confusion: {(true, picked): count}`.
  - `blind_page.py ab --ours DIR --theirs DIR --name-theirs TEXT --sentences FILE --n 20 --seed S --title TEXT --out HTML` and `blind_page.py style --style angry=DIR --style happy=DIR --style sad=DIR --style neutral=DIR --n-per 8 --seed S --title TEXT --out HTML [--ask-natural]`. The style page asks per clip "which delivery?" (happy / sad / angry / neutral) and, with `--ask-natural`, "sounds natural? yes / no". Keys go to `<out>.key.json` next to the page and are not in the page source. A third use, the tag page, is `style` with tag names as keys (`--style laugh=DIR --style none=DIR`).

- [ ] **Step 1: Write the failing tests**

```python
from ttsdata.blindpage import ab_key, tally_ab, style_items, tally_style

def test_ab_key_is_seeded_and_mixed():
    k = ab_key(20, seed=6)
    assert k == ab_key(20, seed=6) and set(k) == {"A", "B"} and len(k) == 20

def test_tally_ab():
    assert tally_ab(["A", "B", "A"], ["A", "A", "same"]) == {"ours": 1, "theirs": 1, "same": 1}

def test_style_items_and_tally():
    by = {s: [f"{s}/{i:04d}.wav" for i in range(10)] for s in ("angry", "sad", "neutral")}
    items = style_items(by, n_per=4, seed=0)
    assert len(items) == 12 and {i["style"] for i in items} == {"angry", "sad", "neutral"}
    idx = [i["wav"].split("/")[1] for i in items]
    assert all(a != b for a, b in zip(idx, idx[1:]))
    t = tally_style(items, [i["style"] for i in items])
    assert t["per_style"]["angry"] == {"correct": 4, "n": 4}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_blindpage.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `ttsdata/blindpage.py`, then `scripts/blind_page.py`**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_blindpage.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Rebuild round 6 from its run dirs as a check** (on astrobot, CPU)

Run: `~/tts/venv-voxcpm/bin/python scripts/blind_page.py ab --ours data/eval/s0_voxcpm2_contB_r6/wavs --theirs data/eval/<the Trendyol run that data/ab_page6.py reads>/wavs --name-theirs Trendyol --sentences data/eval/r6_sentences.txt --n 20 --seed 6 --title "Rebuild check" --out /tmp/check.html`
Expected: a page that opens in a browser, 20 pairs, picks persist on reload, Show result counts match `tally_ab`.

- [ ] **Step 6: Commit**

```bash
git add ttsdata/blindpage.py scripts/blind_page.py tests/test_blindpage.py
git commit -m "Blind A/B and style-identification pages as repo code"
```

---

### Task 3: Stage 2 manifest options (`ttsdata/manifest.py`, `scripts/make_voxcpm_manifest.py`)

**Files:**
- Modify: `ttsdata/manifest.py` (add `pair_neutral_refs`, `folder_rows`)
- Modify: `ttsdata/labels.py` (`allowed_labels` gains `exclude`)
- Modify: `scripts/make_voxcpm_manifest.py` (new flags; the FLEURS replay loop becomes a `folder_rows` call)
- Test: `tests/test_manifest.py`, `tests/test_labels.py`

**Interfaces:**
- Consumes: `labels_of(r)` closure already in the script (labels a row can carry), `pair_references` (kept for stage 1).
- Produces:
  - `pair_neutral_refs(targets: list[dict], pool: list[dict], is_plain: Callable[[dict], bool], ratio: float, seed: int) -> dict[str, str]`: for a seeded `ratio` of targets, a ref from `pool` with the same `speaker_id`, `is_plain(ref)` true, ref id ≠ target id, preferring ≥ 3 s and the same non-null `speaker_hint` (same rules as `pair_references`). Targets whose speaker has no plain clip get no ref. The pool is the whole tier, so a ref may be a clip that is not itself a training row.
  - `folder_rows(rdir: Path, out: Path, want: int, dataset_id: int, seed: int, ref: tuple[str, float] | None = None) -> tuple[list[dict], int]`: rows from a folder of `<id>.wav` + `<id>.txt` (text through `replay_text`), paths relative to `out`, `ref` = (path relative to `out`, duration) added as `ref_audio` / `ref_duration`; returns (rows, skipped).
  - `allowed_labels(styles, tags, min_style, min_tag, exclude: frozenset[str] = frozenset())`: names in `exclude` (style or tag keys, e.g. `angry`, `laugh`) are never allowed, so Task 4 can take a label the base model already does out of training.
  - `combine_events(..., include_unconfirmed: bool | set[str] = False, ...)`: a set names the tags whose `unconfirmed` clips are eligible (`{"cry"}`); `True` keeps today's meaning (all tags). Flag `--include-unconfirmed-tags cry` (the old `--include-unconfirmed` stays).
  - `--narration-styles FILE` (Task 3B output, `{id, style, source}` rows): a further style source merged in `tags_of` like `--auto-styles`; `tier_report.md` counts labels per source (auto, narration, both).
  - New CLI flags: `--styles-exclude angry,...` and `--tags-exclude laugh,...` (both feed `exclude`), `--plain-rows {tier,none}` (default `tier` = today's behaviour; `none` = corpus rows without a written label leave train and val), `--labelled-ref-ratio 0.8` (stage 2: labelled rows get refs from `pair_neutral_refs`), `--anchor-dir DIR`, `--anchor-share 0.5` (anchor rows as a share of the final train), `--anchor-ref data/eval/refs/ref_male_b.wav`, anchor rows `dataset_id: 2`. `tier_report.md` gains a line per new option (anchor rows, labelled rows with a neutral ref, plain rows dropped).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_manifest.py`, reuse its `R` helper)

```python
import numpy as np, soundfile as sf
from ttsdata.manifest import pair_neutral_refs, folder_rows

def test_pair_neutral_refs_takes_plain_same_speaker_only():
    lab = [R(i, spk="a") for i in range(3)]
    pool = lab + [R(10, spk="a"), R(11, spk="a", dur=2.0), R(12, spk="b")]
    plain = lambda r: r["id"] in {"c10", "c11", "c12"}
    pairs = pair_neutral_refs(lab, pool, plain, ratio=1.0, seed=0)
    assert set(pairs) == {"c0", "c1", "c2"} and set(pairs.values()) == {"c10"}   # >= 3 s preferred, never "b"
    assert pair_neutral_refs([R(20, spk="z")], pool + [R(20, spk="z")], plain, 1.0, 0) == {}

def test_folder_rows_with_ref(tmp_path):
    d = tmp_path / "anchor"; d.mkdir()
    for i, t in enumerate(["Merhaba dünya.", "Bugün hava çok güzel."]):
        sf.write(d / f"a{i}.wav", np.zeros(24000 * 2, dtype=np.float32), 24000)
        (d / f"a{i}.txt").write_text(t, encoding="utf-8")
    rows, skipped = folder_rows(d, tmp_path / "out", want=5, dataset_id=2, seed=0, ref=("../refs/b.wav", 6.1))
    assert len(rows) == 2 and skipped == 0 and rows[0]["dataset_id"] == 2
    assert all(r["ref_audio"] == "../refs/b.wav" and r["ref_duration"] == 6.1 for r in rows)
    assert all(validate_row(r) == [] for r in rows)
```

and for unconfirmed clips per tag (append to `tests/test_manifest.py`):

```python
from ttsdata.manifest import combine_events

def test_combine_events_unconfirmed_per_tag():
    units = [R(0), R(1)]
    ev = lambda i, tag: {**R(10 + i), "base": f"c{i}", "tag": tag, "verified": False, "detector": {"reason": "unconfirmed"}}
    events = [ev(0, "cry"), ev(1, "laugh")]
    q = {r["id"]: Q(int(r["id"][1:])) for r in units + events}
    out = combine_events(units, events, q, evidence={"clean"}, include_unconfirmed={"cry"})
    assert sorted(r["id"] for r in out) == ["c1", "c10"]   # the cry clip replaces c0; the laugh clip stays out
```

and in `tests/test_labels.py`:

```python
from ttsdata.labels import allowed_labels

def test_allowed_labels_exclude():
    s, t = allowed_labels(["angry"] * 60 + ["sad"] * 60, ["laugh"] * 400, 50, 300, exclude=frozenset({"angry", "laugh"}))
    assert s == {"sad"} and t == set()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_manifest.py tests/test_labels.py -v`
Expected: the four new tests FAIL (`ImportError`, `TypeError: unexpected keyword 'exclude'`, the laugh clip let in); the old ones PASS.

- [ ] **Step 3: Implement both functions and the flags**

With `--plain-rows none`, `labels_of` decides plainness after `allowed_labels` (a row whose label is not written counts as plain). The `--label-repeat` oversampling still applies to labelled rows; anchors are counted after oversampling, `want = round(anchor_share / (1 - anchor_share) * len(train))`. FLEURS replay keeps working through `folder_rows(..., dataset_id=1)`; stage 1 output with default flags must not change (Step 4 checks it).

- [ ] **Step 4: Run the tests, then check stage 1 output did not change**

Before committing, save the committed script: `git show HEAD:scripts/make_voxcpm_manifest.py > <scratch>/old_manifest.py`. Build stage 1 locally with both (`PYTHONPATH=. .venv/bin/python <script> --root data/dataset --out <scratch>/{old,new} --wavs wavs --max-hours 60`; the scored held-out file in `data/eval/` is only written when missing, so it is not touched) and run `diff <scratch>/old/train.jsonl <scratch>/new/train.jsonl`.
Expected: `pytest tests/ -q` all PASS; no diff.

- [ ] **Step 5: Commit**

```bash
git add ttsdata/manifest.py ttsdata/labels.py scripts/make_voxcpm_manifest.py tests/test_manifest.py tests/test_labels.py
git commit -m "Stage 2 manifest: labelled rows only, neutral refs for labelled rows, anchor rows from a folder"
```

---

### Task 3A: Label-first cleaning (GPU ~10 h, ask first)

Every training clip must be BS-Roformer-separated (speech: then Sidon-restored); so far only clips picked for the stage 1 slices were. This pass cleans the labelled ones.

**Files:**
- Create: `scripts/server_label_clean.sh` (copy of `scripts/server_more_clean.sh`; markers `data/logs/lclean_<stage>_done`, log `data/logs/lclean.log`; the `label_` prefix is taken by `server_label_stages.sh`)

**Interfaces:**
- Consumes: Task 3 flags; `scripts/separate_vocals.py`, `cut_context.py`, `sidon_restore.py`, `sidon_wordcheck.py`, `audit_quality.py` exactly as `server_more_clean.sh` calls them.
- Produces: `wavs_clean` / `wavs_sidon` / `restoration.jsonl` rows for every labelled clip in `data/dataset`, `data/dataset_long`; `wavs_clean` for every verified or unconfirmed event clip and its base in `data/dataset_events`; the selection manifest `data/voxcpm_stage2_rawlabels/`.

- [ ] **Step 1: Write the driver.** The `rawslice` stage becomes a raw-audio stage 2 build that takes labelled clips only:

```bash
"$PY" scripts/make_voxcpm_manifest.py --stage 2 --out data/voxcpm_stage2_rawlabels --wavs wavs --long-wavs wavs \
  --events-wavs wavs --auto-styles --auto-conf angry=0.70,sad=0.85,happy=0.99 --min-tag 150 \
  --include-unconfirmed-tags cry,laugh,breath --plain-rows none --label-first --label-quota 1000000 \
  --label-max-share 1.0 --label-repeat 1 --replay-share 0 --max-hours 400 --max-series-share 1.0
```
(unconfirmed laughs and breaths are cleaned too, so a later yes needs no second pass; `--narration-styles` joins when Task 3B has run). Then the same per-root stages as `server_more_clean.sh` for `dataset_long` and `dataset`, plus `sep_dataset_events` (separation only, no Sidon). Every step skips clips already done.

- [ ] **Step 2: Dry run on the Mac** against the local 18 h build with `--max-hours 5`: the stages run end to end on CPU (`SEP_JOBS=1 SIDON_JOBS=1`, Sidon may be skipped locally with a stub note in the log).

- [ ] **Step 3: On astrobot**: wait for `run5b`'s cleaning stages (`data/logs/more_clean_done`, or ask the user to stop its training part), ask the user, then `tmux new -d -s lclean 'scripts/server_label_clean.sh >> data/logs/lclean.log 2>&1'`.
Expected: `tier_report.md` of `data/voxcpm_stage2_rawlabels` lists the labelled hours per label before cleaning; after the run, a restored-only build (Task 5 Step 3) has ≥ 15 h per style and ≥ 250 cry clips (≥ 300 with the unconfirmed ones). Record both in `stage2_results.md`.

- [ ] **Step 4: Commit** `scripts/server_label_clean.sh`.

---

### Task 3D: Text recovery (keep good subtitles; a second recogniser breaks ties)

Most emotional clips never reach the tier, and not because of audio: the music / silence / sound-flag gates drop < 1% of them. Of 62,342 angry single clips, 45,057 are shorter than 3 s and 4,706 fail the 5% Whisper-vs-subtitle gate; 124,246 lines (64 h) were rejected before the dataset for a > 20% mismatch and never scored for emotion. The user's listening check (2026-11-01, 20 clips the text check drops) found the subtitle mostly right where Whisper differs. Rule agreed with the user:

| Whisper vs subtitle | text |
|---|---|
| ≤ 20% | the subtitle (stage 2 builds pass `--max-cer 0.20`) |
| > 20% (`high_cer` rejects) | second recogniser (another model family than Whisper) votes: agrees with the subtitle (≤ 5%) → subtitle; agrees with Whisper and not the subtitle → what is spoken; neither → drop |
| `multi_speaker`, `unaligned`, `no_speech`, `credits` | dropped |

Labelled lines of 1.5–3 s join the tier (`--label-min-dur 1.5`); unlabelled rows keep 3 s. Before training on them, a listening check of 30 short clips per style (Task 5) confirms the emotion labels hold on short clips. **Result (2026-11-01): they do not** (the listener heard the automatic label on 7/20 angry, 5/19 happy, 9/20 sad short clips; most sound neutral), so automatic labels keep the 3 s minimum and the selection uses `--max-cer 0.20` only.

**Files:**
- Modify: `ttsdata/manifest.py` (`keep_short_labelled`), `scripts/make_voxcpm_manifest.py` (`--label-min-dur`), `scripts/server_label_clean.sh` (selection flags)
- Create: `ttsdata/textvote.py`, `scripts/second_asr.py`, `scripts/recover_clips.py`; tests `tests/test_textvote.py`

**Interfaces:**
- `keep_short_labelled(rows, labels_of, plain_min_dur=3.0) -> list[dict]`: rows ≥ `plain_min_dur`, plus shorter rows with a label.
- `--label-min-dur S` (stage 2; 0 < S ≤ 3): the duration gate's minimum becomes S, then unlabelled rows under 3 s leave the tier; default off (builds unchanged).
- `vote(subtitle: str, whisper: str, asr2: str, agree: float = 0.05) -> str`: "subtitle" | "spoken" | "drop", CERs on `for_cer` text.
- `second_asr.py bench|run`: `bench` scores candidate models on clips whose subtitle is trusted (Whisper CER 0) and on `high_cer` rejects; `run --ids FILE` writes `data/dataset/asr2.jsonl` `{id, model, text}` (shards, resumable).
- `recover_clips.py`: applies `vote` to `high_cer` rejects ≥ 1.5 s; winners go to a new root `data/dataset_recovered` (metadata rows with the training text, trims from the segment results, speaker = nearest existing ECAPA cluster centre above 0.70 else dropped), then emotions, audit and soundtags run there. Existing roots stay untouched, so stage 1 builds stay byte-identical; the manifest reads it with `--recovered-root`.

- [x] **Step 1 (Mac):** TDD `keep_short_labelled` and `--label-min-dur`; stage 1 build byte-identical.
- [x] **Step 2:** the `lclean` selection takes `--max-cer 0.20 --label-min-dur 1.5` before it starts; preview counts per label in `stage2_results.md`.
- [x] **Step 3 (GPU, small):** `second_asr.py bench` on 300 trusted clips + 300 `high_cer` rejects; candidates `facebook/seamless-m4t-v2-large`, `facebook/mms-1b-all` (adapter `tur`), Meta Omnilingual ASR if it installs. Pick the lowest CER on the trusted set; record agreement rates (with subtitle / with Whisper / neither) and the projected recovery.
- [ ] ~~**Step 4:**~~ **Dropped (user, 2026-11-01):** the bench rescues ~7% of the rejects at a 5% bar (~4.7k of 67k lines), ~20% at 15% with up to 15% recogniser errors in the text; small and noisy next to steps 1–2, and it needs a new data root. `vote` and `second_asr.py` stay in the repo. Was: TDD `vote`; `second_asr.py run` on the `high_cer` rejects ≥ 1.5 s (~67k); `recover_clips.py`; emotions + audit on `dataset_recovered`; a second `lclean` pass cleans its labelled clips (finished clips are skipped).

Expected: per style, the restored-only build of Task 5 gains clips from short lines, from the 5–20% band and from recovered lines; numbers per source in `stage2_results.md`.

---

### Task 3B: Narrator emotion labels (Engelsiz audio description)

The TRT narrator says what characters feel ("Asya üzgün haliyle Ömer'in yanına..."). Linked to that character's next lines, this is a label made by a human describer, independent of emotion2vec, and the only source for crying-while-talking, fear and surprise.

**Files:**
- Create: `ttsdata/narration.py` (pure: lexicon, parsing, linking)
- Create: `scripts/transcribe_narration.py` (GPU: Whisper on the gaps between subtitle lines)
- Create: `scripts/narration_labels.py` (CPU: parse + link → `narration_styles.jsonl`; calibration page via `blind_page.py style`)
- Test: `tests/test_narration.py`

**Interfaces:**
- `EMOTION_STEMS: dict[str, tuple[str, ...]]` = angry `("sinir", "öfke", "kızgın", "hiddet", "bağır", "terslen")`, sad `("ağla", "gözyaş", "hüzün", "üzgün", "üzül", "hıçkır", "buruk")`, happy `("gülümse", "gülerek", "gülüyor", "sevin", "neşe", "mutlu", "keyif", "kahkaha")`, scared `("kork", "dehşet", "panik", "ürk")`, surprised `("şaşkın", "şaşır", "hayretle")`. A separate `crying` class for `("ağlıyor", "ağlayarak", "hıçkır")` so crying-while-talking can become its own style if it has the volume.
- `parse_description(text: str) -> list[dict]`: `{"name": str | None, "emotion": str, "word": str}` per emotion word; `name` = the capitalised word(s) before it in the same clause with the Turkish case suffix after an apostrophe removed ("Ayşe'nin" → "Ayşe"); negated forms (stem + `ma/me` + `dan/den/k/y/dı/di`, e.g. "gülümsemeden", "ağlamamak") yield nothing.
- `name_clusters(rows: list[dict], min_lines: int = 5, min_share: float = 0.6) -> dict[tuple[str, str], str]`: (series, name) → `speaker_id` when that name's `speaker_hint` lines fall mostly in one cluster.
- `link_lines(desc: dict, lines: list[dict], cluster: str | None, before: float = 2.0, after: float = 20.0) -> list[str]`: ids of lines of `cluster` that start within [desc end − `before`, desc end + `after`] and before the next description of the same character; with no name, only the first line starting within 5 s (marked `source: "narration_noname"`).
- `transcribe_narration.py --source trt [--limit-episodes N]`: per episode, gaps ≥ 1.5 s between subtitle cues (`subs.srt`), faster-whisper `deepdml/faster-whisper-large-v3-turbo-ct2`, `language="tr"`, word timestamps, on the original audio; drops `ttsdata.eventverify.HALLUCINATION_RE` matches, "izlediğiniz için teşekkür", "sesli betimleme". Output `data/dataset/narration.jsonl` `{uid, start, end, text}`, resumable per episode.
- `narration_labels.py` → `data/dataset/narration_styles.jsonl` `{id, style, source, desc, dt}` and `data/dataset/NARRATION.md` (descriptions per emotion, linked lines per emotion, agreement with emotion2vec+ per class).

- [x] **Step 1: Write the failing tests** (`tests/test_narration.py`)

```python
from ttsdata.narration import parse_description, name_clusters, link_lines

def test_parse_description():
    assert parse_description("Şivenaz zindanda sinirle dört dönerken") == [{"name": "Şivenaz", "emotion": "angry", "word": "sinirle"}]
    assert parse_description("Ayşe'nin saçlarını okşarken ağlıyor.")[0]["name"] == "Ayşe"
    assert parse_description("Sade Nur sevinçle gülümsedi.")[0] == {"name": "Sade Nur", "emotion": "happy", "word": "sevinçle"}
    assert parse_description("Ağlamamak için dudağını ısırdı.") == []
    assert parse_description("Kapıyı açıp içeri girdi.") == []

def test_name_clusters_majority():
    rows = [{"series": "S", "speaker_hint": "Ayşe", "speaker_id": "S_spk1"}] * 8 + [{"series": "S", "speaker_hint": "Ayşe", "speaker_id": "S_spk2"}] * 2
    assert name_clusters(rows) == {("S", "Ayşe"): "S_spk1"}
    assert name_clusters(rows[:4]) == {}                                  # too few named lines

def test_link_lines_window_and_cluster():
    desc = {"name": "Ayşe", "end": 100.0}
    lines = [{"id": "a", "start": 99.0, "speaker_id": "S_spk1"}, {"id": "b", "start": 110.0, "speaker_id": "S_spk2"},
             {"id": "c", "start": 115.0, "speaker_id": "S_spk1"}, {"id": "d", "start": 125.0, "speaker_id": "S_spk1"}]
    assert link_lines(desc, lines, "S_spk1") == ["a", "c"]
    assert link_lines({"name": None, "end": 100.0}, lines, None) == []     # no line starts within 5 s after
```

- [x] **Step 2: Run, implement `ttsdata/narration.py`, run again** (2026-11-02; also `scripts/transcribe_narration.py` and
  `scripts/narration_labels.py`, smoke-tested on one local episode and on synthetic data). Rulings: the happy stem is
  `gülüms` ("gülümse-" narrows to "gülümsüyor", the commonest form); a word matching both a crying form and a sad
  stem is crying; one row per emotion per sentence; a capitalised word inside a sentence is a name, never an emotion
  ("Sevinç", "Neşe"); `NOT_EMOTION` = korkuluk, bağırsak; negation also covers -mıyor, -madan, -mamak, -sız and a
  stem vowel before -ma; spans with "betimle" / "izlediğiniz için" (the describer's intro / outro) are dropped. Gaps
  come from speech cues only (`clean_cue`), lose 0.2 s next to a cue, and all gaps of an episode are transcribed in
  one batched faster-whisper pass (VAD on), words handed back with `asr.split_pack_result`. Names resolve case-folded,
  full name first, else its first word; links go to `data/dataset` lines only (not `dataset_long` merges).

Run: `.venv/bin/python -m pytest tests/test_narration.py -v` (FAIL `ModuleNotFoundError`, then 3 PASS). Episode times: subtitle lines and clips carry episode start times (`orig_start` style fields in `metadata.jsonl`); use the same field `extend-events` uses.

- [ ] **Step 3: Pilot on 20 TRT episodes** (GPU, ask first): `transcribe_narration.py --limit-episodes 20`, then `narration_labels.py`.
Expected in `NARRATION.md`: narrator seconds per episode, Whisper real-time factor, descriptions per emotion, linked lines per emotion, share of descriptions with a resolvable name. Extrapolate to 1,017 TRT episodes and write it in `stage2_results.md`. Stop and show the user if fewer than ~2,000 linked lines per class are projected.

- [ ] **Step 4: Full run** (GPU ~3–5 h by the pilot's rate, ask first; tmux `narr`, marker `data/logs/narration_done`).

- [ ] **Step 5: Calibrate by ear.** Calibration page of 40 linked clips per class (angry, sad, happy, and crying / scared / surprised if each has ≥ 1,000 linked lines), half where emotion2vec+ agrees at confidence ≥ 0.5, half where it does not; the user rates the delivery as in the 2026-10-27 calibration (`ttsdata.emotion.RATING_CHOICES`). Score with `ttsdata.emotion.score_ratings`.
Decision per class and source: narrator alone, narrator + emotion2vec+ ≥ 0.5, emotion2vec+ alone at the calibrated cut. A source is used when its rated precision ≥ 0.76 (the spec's target, as in the 2026-10-27 calibration). A new style (crying, scared, surprised) enters stage 2 only with precision ≥ 0.76 and ≥ 3,000 clips; otherwise it waits for a later run.

- [ ] **Step 6: Commit** `ttsdata/narration.py`, both scripts, `tests/test_narration.py`, the ratings sheet under `calibration/`, `stage2_results.md`.

---

### Task 3C: Event clips: same person, no narrator

Cry check 1 (2026-11-01, `data/listening_sample/stage2_cry_check.html`): the user heard crying in 20 of 21 rated unconfirmed clips and 3 of 3 rated verified ones, so unconfirmed cries go in. The user also heard two faults that would teach `[ağlama]` the wrong sound: in some clips the cry belongs to another person than the one speaking the line (dialogue scene), and in some the female audio-description narrator talks over the moment. Both apply to laughs and breaths too. This task flags them for every event clip.

**Files:**
- Create: `ttsdata/eventclean.py` (pure checks)
- Create: `scripts/check_events.py` (CPU: F0 per extension, writes `data/dataset_events/event_checks.jsonl`)
- Modify: `scripts/make_voxcpm_manifest.py` (`--event-checks FILE`: flagged event clips leave; their base clip returns as for an unwritten tag, via `restore_bases`)
- Test: `tests/test_eventclean.py`

**Interfaces:**
- Consumes: event rows (`event: {start, end, who}`, `uid`, `speaker_id`, `series`), Task 3B `narration.jsonl` spans and `name_clusters`, `data/dataset/speaker_gender.jsonl`, `ttsdata.voicegender.clip_f0`.
- Produces:
  - `narrator_overlap(event: dict, spans: list[tuple[float, float]], min_sec: float = 0.3) -> bool`: the extension (`event.start`–`event.end`, episode time) overlaps transcribed narrator speech by ≥ `min_sec`.
  - `who_conflict(who: str | None, series: str, line_speaker: str, clusters: dict[tuple[str, str], str]) -> bool`: the subtitle names the sound's maker and that name maps to another cluster than the line's speaker. Unknown names never conflict.
  - `pitch_conflict(ext_f0: float | None, line_gender: str | None, male_line_max: float = 260.0, female_line_min: float = 140.0) -> bool`: male line with an extension median F0 ≥ 260 Hz, or female line with ≤ 140 Hz (crying raises pitch, so the bars sit well outside `voicegender`'s 150 / 175 Hz).
  - `event_checks.jsonl` rows `{id, narrator, who_conflict, pitch_conflict, ext_f0}`; a row is flagged when any check is true.

- [ ] **Step 1: Write the failing tests** (`tests/test_eventclean.py`)

```python
from ttsdata.eventclean import narrator_overlap, who_conflict, pitch_conflict

def test_narrator_overlap():
    ev = {"start": 100.0, "end": 102.0}
    assert narrator_overlap(ev, [(101.5, 104.0)])
    assert not narrator_overlap(ev, [(101.9, 104.0)])       # 0.1 s: a breath of overlap, not speech over the cry
    assert not narrator_overlap(ev, [])

def test_who_conflict():
    cl = {("S", "Afra"): "S_spk3"}
    assert who_conflict("Afra", "S", "S_spk7", cl)
    assert not who_conflict("Afra", "S", "S_spk3", cl)
    assert not who_conflict("Kerem", "S", "S_spk7", cl)      # name not mapped: no evidence either way
    assert not who_conflict(None, "S", "S_spk7", cl)

def test_pitch_conflict():
    assert pitch_conflict(300.0, "male") and pitch_conflict(120.0, "female")
    assert not pitch_conflict(230.0, "male")                 # a man crying can sound high
    assert not pitch_conflict(None, "male") and not pitch_conflict(300.0, None)
```

- [ ] **Step 2: Run, implement, run again**

Run: `.venv/bin/python -m pytest tests/test_eventclean.py -v` (FAIL `ModuleNotFoundError`, then 3 PASS); add a manifest test that a flagged event clip hands its place back to its base.

- [ ] **Step 3: Run `scripts/check_events.py`** on astrobot after Task 3B Step 4 (CPU; F0 on the extension part of each event clip's `wavs_clean` audio). Expected: counts flagged per check and per tag in `stage2_results.md`.

- [ ] **Step 4: Cry check 2** (blind page, one-off builder like `data/cry_page.py` or `blind_page.py style`): 30 cry clips that pass the checks + 10 flagged ones, mixed; answers "same person crying / another person crying / narrator talks / nothing". Gate (my call): among passing clips ≥ 24 of 30 "same person crying" and ≤ 1 "narrator talks". If it fails, tighten the bars (`min_sec`, F0) once and re-check; a second failure leaves `[ağlama]` out of run 2a.

- [ ] **Step 5: Commit** `ttsdata/eventclean.py`, `scripts/check_events.py`, the manifest change and tests.

---

### Task 4: Base-model probe (GPU, ~1 h, ask first)

What does untrained VoxCPM2 already do? Each style and tag is tried in four setups, all with clip B's voice and cfg 3.0:

| setup | prompt / clip | how |
|---|---|---|
| P1 continue, Turkish | S1 + Turkish prefix | `--prompt-wav B --ref-wav B --prompt "(öfkeyle)"` |
| P2 clone, Turkish | clone only + Turkish prefix | `--ref-wav B --prompt "(öfkeyle)"` |
| P3 clone, English | clone only + English prefix | `--ref-wav B --prompt "(An angry voice)"`; happy `"(A happy, cheerful voice)"`, sad `"(A sad voice)"` |
| P4 styled clip | continue from a styled clip, clone clip B | `--prompt-wav <style clip> --ref-wav B --prompt ''` |

Plain baseline P0 = S1 on the same sentences. Tags are tried in P1 and P2 with the Turkish tag inline, and in P2 with `[laughter]` / `[breath]` / `[crying]` in place of the Turkish tag.

**Files:**
- Create: `data/eval/style_neutral.txt` (Freya lines 120–139: emotion-free text, so any style must come from the voice)
- Create: `data/eval/style_matched_{angry,happy,sad}.txt` (10 texts each from `data/voxcpm_stage2/val.jsonl` rows carrying that prefix, label stripped with `strip_labels`: text that fits the style; the current build uses the same calibrated cuts and the same val split as the Task 5 rebuild)
- Create: `data/eval/tags_{laugh,breath,cry}.txt` (Freya lines 140–169, 10 per tag, tag at start for 5 and at end for 5, via `ttsdata.labels.tag_text`)
- Create: `data/eval/refs/style_{angry,happy,sad}.wav` + `.txt` (for P4: a male restored clip per style, emotion2vec+ confidence above the calibrated cut, PESQ ≥ 3.5, 4–8 s; the user listens to the three before use)
- Create: `scripts/probe_stage2.sh` (loops over setups × labels × sets with `eval_tts.py all` then `score_styles.py`; runs named `s2p_<setup>_<label>_<set>`; done marker `data/logs/s2_probe_done`)
- Create: `docs/superpowers/plans/stage2_results.md` (probe table and decision)

- [x] **Step 1: Write the sentence files and pick the P4 clips** (Mac for Freya text, astrobot for the matched text and the clips).
  2026-11-02: `data/eval/` is gitignored, so the sentence files are added with `git add -f`; also `tags_plain.txt` (P0 on
  the tag sentences) and `tags_<tag>_en.txt`. P4: three male candidates per style (one per speaker) in
  `data/listening_sample/style_refs.html`; the user's pick is copied to `data/eval/refs/style_<label>.wav`, and
  `probe_stage2.sh` skips P4 until then.

- [x] **Step 2: Ask the user for the GPU, check `tmux ls`, launch `scripts/probe_stage2.sh` in tmux `s2probe`** (launched
  2026-11-02 00:50 after the user's go-ahead; P0-P3 and tags first)

Expected: about 1,000 short clips; `style_summary.md` rows for every run.

- [x] **Step 3: Build a probe listening page** (2026-11-02: style pages for P2, P3, P4 only, since P1 reads the prefix aloud in
  7-20 of 20 sentences; tag pages for P1, P2, P2en; numbers in `stage2_results.md`) with `blind_page.py style` per setup (8 neutral-text clips per style plus 8 P0 clips as "neutral"), and one tag page per setup (`laugh`, `breath`, `cry`, `none`). Send the pages to the user.

- [x] **Step 4: Record and decide** in `stage2_results.md` (2026-11-02: no label leaves the training set; styled
  inference mode P2; emotion2vec+ hit rates on synthetic speech disagree with the listener, so they pre-filter only):

| per label | measure |
|---|---|
| style, per setup | emotion2vec+ hit rate on neutral text and on matched text; neutral rate; listener correct of 8; WER; voice cos to P0; read aloud |
| tag, per setup | laugh realised (Whisper); listener heard the sound of 10; tag word read aloud |

Decision rule: a label whose best setup already meets its Task 7 gate (style: listener ≥ 6/8 and hit rate gate; tag: listener ≥ 7/10) **leaves the training set** (Task 5 builds with `--styles-exclude` / `--tags-exclude` for it) and its inference recipe is recorded. The setup with the best style results among P1/P2 (the ones a LoRA can improve) becomes the **styled inference mode** for Task 6. If no label is missing, stage 2 ends here with no training: write that up and stop.

- [ ] **Step 5: Commit** sentence files, `probe_stage2.sh`, `stage2_results.md` (not the wavs).

---

### Task 5: Training data: Sidon check, anchor set, manifest rebuild (GPU, ask first)

**Files:**
- Create: `scripts/make_anchor_set.py` (subcommands `sentences` and `export`)
- Create: `ttsdata/anchors.py` (pure: `pick_sentences`, `anchor_keep`)
- Test: `tests/test_anchors.py`

**Interfaces:**
- `pick_sentences(rows: list[dict], exclude: set[str], n: int, seed: int) -> list[str]`: rows are tier rows (already through the quality gates); returns `normalized_text` of train-split, unlabelled clips 3–20 s, no `text_flags`, not in `exclude` (every line of every `data/eval/*.txt`), deduplicated, seeded sample of `n`.
- `anchor_keep(score: dict, min_ref_cos: float = 0.75) -> bool`: keep when `wer == 0`, `read_aloud == []`, `spk_cos` (to the run centre) ≥ `min_ref_cos`, and `utmos` is not None. (High-voice drift shows up as low `spk_cos`.)
- `make_anchor_set.py sentences --n N --out data/eval/anchor_sentences.txt`; `make_anchor_set.py export --run s2_anchor --sentences data/eval/anchor_sentences.txt --out data/anchor_base` → `<k>.wav` (resampled to 24 kHz, −20 LUFS via `ttsdata.build._loudness_normalize`) + `<k>.txt`, and a printed keep rate.

- [ ] **Step 1: Write the failing tests** (`tests/test_anchors.py`)

```python
from ttsdata.anchors import pick_sentences, anchor_keep

def test_pick_sentences_filters_and_excludes():
    rows = [{"normalized_text": f"Cümle {w}.", "duration": 4.0, "split": "train", "labelled": False} for w in ("bir", "iki", "üç")]
    rows += [{"normalized_text": "Kısa.", "duration": 2.0, "split": "train", "labelled": False},
             {"normalized_text": "Val.", "duration": 4.0, "split": "val", "labelled": False},
             {"normalized_text": "Öfkeli.", "duration": 4.0, "split": "train", "labelled": True}]
    out = pick_sentences(rows, exclude={"Cümle iki."}, n=10, seed=0)
    assert sorted(out) == ["Cümle bir.", "Cümle üç."]

def test_anchor_keep():
    ok = {"wer": 0.0, "read_aloud": [], "spk_cos": 0.82, "utmos": 3.1}
    assert anchor_keep(ok)
    assert not anchor_keep({**ok, "wer": 0.1}) and not anchor_keep({**ok, "spk_cos": 0.6})
```

- [ ] **Step 2: Run, implement, run again**

Run: `.venv/bin/python -m pytest tests/test_anchors.py -v` (FAIL `ModuleNotFoundError`), implement `ttsdata/anchors.py` and the script, run again (2 PASS). Rows passed to `pick_sentences` get `labelled` from the stage 2 tier (the script computes it with the same `labels_of` logic; factor it out of `make_voxcpm_manifest.py` into `ttsdata/manifest.py` if needed rather than copying it).

- [ ] **Step 3: Rebuild `data/voxcpm_stage2`** on astrobot (CPU; keep the old build as `data/voxcpm_stage2_prev`)

```bash
cd ~/voice-data && mv data/voxcpm_stage2 data/voxcpm_stage2_prev
~/tts/venv-voxcpm/bin/python scripts/make_voxcpm_manifest.py --stage 2 --out data/voxcpm_stage2 \
  --wavs wavs_sidon --long-wavs wavs_sidon --restored-only --events-wavs wavs_clean \
  --auto-styles --auto-conf angry=0.70,sad=0.85,happy=0.99 --min-tag 150 \
  [--narration-styles data/dataset/narration_styles.jsonl] --include-unconfirmed-tags cry \
  --event-checks data/dataset_events/event_checks.jsonl \
  --plain-rows none --labelled-ref-ratio 0.8 --label-first --label-quota 6000 --label-max-share 1.0 \
  --label-repeat 1 --replay-share 0 --max-hours 80 --anchor-share 0 [--styles-exclude/--tags-exclude from Task 4]
```
Expected: `tier_report.md` lists per label the clips in tier, in slice and written; each style at its 6,000 quota (≈ 7–9 h), labelled train rows ≈ 20k–24k; every train row labelled; no FLEURS rows. Write the label table into `stage2_results.md`. Quota 6,000 keeps the styles balanced and the anchor synthesis near a night's work; raise it in a later run only if a style misses its gate narrowly. If `--restored-only` leaves any style under 3,000 clips, stop and ask the user.

Unconfirmed cries are in: cry check 1 heard crying in 20 of 21 rated unconfirmed clips (bar 0.8). Event clips flagged by Task 3C (another person's cry, narrator over the sound) are out for every tag.

- [ ] **Step 4: Does Sidon keep the style?** (GPU, ~15 min, ask first)

For 200 written clips per style, score both `wavs_clean` and `wavs_sidon` copies: `score_styles.py --wavs <folder of symlinks> --target <style> --run s2_sidoncheck_<style>_<wavs>`. Record hit rates. If the Sidon hit rate is under 0.8 × the `wavs_clean` hit rate for a style, stop and ask the user before training (Sidon is flattening that style).

- [ ] **Step 5: Synthesise the anchor set** (GPU, ask first; tmux `s2anchor`, marker `data/logs/s2_anchor_done`)

N = 1.25 × labelled train rows from Step 3 (expect ~80 % to pass `anchor_keep`). Synthesise with S1 exactly (`eval_tts.py all --system voxcpm2 ... --run s2_anchor --sentences data/eval/anchor_sentences.txt --utmos 1 --spk 1`), first `--limit 200` to measure the real-time factor and write the full-run estimate in `stage2_results.md`, then the rest (synth resumes). Then `make_anchor_set.py export`.
Expected: `data/anchor_base/` with ≥ labelled-train-rows clips; keep rate printed.

- [ ] **Step 6: Final build with anchors**

Same command as Step 3 plus `--anchor-dir data/anchor_base --anchor-share 0.5 --anchor-ref data/eval/refs/ref_male_b.wav`.
Expected: train = labelled rows + equal number of anchor rows (`dataset_id: 2`, all with clip B as ref); `validate_row` passes; listen to 5 random labelled rows and 5 anchors via `data/listening_sample/stage2_manifest.html` (built with `blind_page.py style --style <label>=...`, non-blind, keys shown) before training.

- [ ] **Step 7: Commit** `ttsdata/anchors.py`, `scripts/make_anchor_set.py`, `tests/test_anchors.py`, `data/eval/anchor_sentences.txt`, `stage2_results.md` (manifests and wavs stay on the server).

---

### Task 6: Run 2a, LM-only LoRA from base, and checkpoint scoring (GPU, ask first)

**Files:**
- Create: `training/voxcpm/stage2_style_lm.yaml`
- Create: `scripts/train_stage2.sh` (tmux launcher, same shape as `scripts/train_stage1.sh`)
- Create: `scripts/score_stage2.sh` (per checkpoint and scale)

**Interfaces:**
- Consumes: `data/voxcpm_stage2/{train,val}.jsonl` (Task 5), styled inference mode from Task 4, `eval_tts.py --lora-scale S,0` (Task 1).
- Produces: `~/tts/checkpoints/stage2_v2a/step_*`, runs `s2a_step<N>_x<S>_<set>`, rows in `summary.md` and `style_summary.md`.

- [ ] **Step 1: Write `training/voxcpm/stage2_style_lm.yaml`**

Copy `training/voxcpm/stage1_run5a_lm.yaml` (header comment: what this run is, the manifest command from Task 5 Step 6, the step count) and set: `train_manifest` / `val_manifest` → `data/voxcpm_stage2`, `save_path: /home/ibrahim/tts/checkpoints/stage2_v2a`, `tensorboard: /home/ibrahim/tts/logs/stage2_v2a`, `lora: {enable_lm: true, enable_dit: false, enable_proj: false, r: 32, alpha: 32, dropout: 0.0}`, `learning_rate: 0.0001`, `warmup_steps: 100`, batch 4 × accum 4, `num_iters` = `max_steps` = 2 epochs = `2 * ceil(lines(train.jsonl) / 16)` (write the number), `save_interval: 200`, `valid_interval: 200`. Rank 32 (half of stage 1) because the adapter has less to learn: styles and four tags, not a language.

- [ ] **Step 2: Ask for the GPU, `tmux ls`, launch** `tmux new-session -d -s stage2a 'bash ~/voice-data/scripts/train_stage2.sh'`; check `nvidia-smi` and the log after 5 minutes.
Expected: loss lines every 10 steps, no OOM, ~0.6 s/step (≈ 30 min for 3,000 steps).

- [ ] **Step 3: Write and run `scripts/score_stage2.sh STEP SCALE`** for steps 400, 800, 1200, … , end at scale 1.0, marker `data/logs/score_s2a_done`

Per call, with `--system voxcpm2-lora --lora .../step_<STEP> --lora-scale <SCALE>,0 --cfg 3.0`:
1. plain: Freya 0–39 in S1 (`--prompt-wav B --ref-wav B --prompt ''`), `eval_tts.py all`, then `score_styles.py --target neutral --voice-ref-run s0_voxcpm2_contB_freya40` (stage 1's untrained clip B run on the same 40 lines; regenerate it if missing);
2. styles: `style_neutral.txt` and `style_matched_<style>.txt` per style in the Task 4 mode with the Turkish prefix as `--prompt`;
3. tags: `tags_<tag>.txt` per tag in the Task 4 mode.
Then also PESQ / UTMOS p10 / CPPS / syl/s for the plain set with the stage 1 probe scripts (`data/*.py` on astrobot; copy the one used for the round 4 table into `scripts/` if it is not there yet).

- [ ] **Step 4: Pick the candidate and sweep the strength**

Candidate = the earliest checkpoint that meets the style and tag number gates (Task 7 table) for the labels it trains. Score it again at `--lora-scale` 0.5 and 0.75. Pick the **lowest scale** that still meets the number gates. Record all rows in `stage2_results.md`.

- [ ] **Step 5: Commit** the config, both scripts and `stage2_results.md`.

If no checkpoint meets the number gates: one retry (run 2b) with `r: 64, alpha: 64` and `--label-repeat 2`, same anchors. A second failure stops stage 2 for a talk with the user.

---

### Task 7: Exit gates (listening + numbers)

All pages built with `scripts/blind_page.py`, sentences never heard before, keys kept out of the page.

| gate | measure | pass |
|---|---|---|
| **Plain regression** (user's rule) | blind A/B, 20 plain Freya sentences (200–219), candidate in S1 vs untrained in S1 | untrained preferred ≤ candidate preferred (ties fine) |
| Plain numbers | Freya 0–39 in S1 vs `s0_voxcpm2_contB_freya40` | WER ≤ untrained + 0.5 pt; PESQ p10 and UTMOS p10 ≥ untrained − 0.1; syl/s within ±0.5 of untrained; read aloud 0 |
| Style by ear | style page: 8 neutral-text clips per trained style + 8 plain clips, listener picks the delivery | each trained style ≥ 6/8 correct; plain clips picked "neutral" ≥ 6/8; "natural" ≥ 6/8 per style |
| Style by numbers | emotion2vec+ hit rate on `style_neutral.txt` | angry ≥ 0.64, sad ≥ 0.42 (CosyVoice 3 text-unrelated figures, spec), happy ≥ 0.42; each ≥ base P1/P2 + 0.20 |
| Voice kept | ECAPA cosine of styled clips to the plain untrained run centre | mean ≥ 0.70 |
| Styled intelligibility | WER on style and tag sets, labels stripped | ≤ 3 % |
| Tags | tag page: 10 clips per tag + 10 untagged | `[gülme]` heard ≥ 7/10 and Whisper laugh realised ≥ 0.7; `[nefes]` ≥ 7/10; `[ağlama]` ≥ 6/10; untagged clips with a heard sound ≤ 1/10; tag word read aloud 0 |

- [ ] **Step 1: Build and send the three pages** (`stage2_ab_blind1.html`, `stage2_style1.html`, `stage2_tags1.html` in `data/listening_sample/`).

- [ ] **Step 2: Record the user's answers** with `tally_ab` / `tally_style` in `stage2_results.md` (one section per round, as in stage 1).

- [ ] **Step 3: Decide**

- All gates met → stage 2 passes with this checkpoint and scale; write the inference recipe at the top of `stage2_results.md`.
- Only the plain regression fails → deploy rule "adapter only for text that carries a label": plain text goes to the untrained model (`--lora-scale 0,0` is not exact, so route plain text to `--system voxcpm2`). Ask the user whether that counts as a pass.
- A single tag fails (likely `[ağlama]`) → stage 2 passes without that tag; the tag goes back to metadata and is named in the results as stage 3 work. More cry clips can only come from the cue funnel (standalone cues joined to a line further away, or a cry detector), not from more cleaning: Task 3A already cleans every usable cry clip.
- Styles fail by ear → back to Task 6 retry rule.

- [ ] **Step 4: Commit** `stage2_results.md` and push to `main`.

---

## Self-review notes

- Spec coverage: labelled subset with Turkish prefixes and tags (Task 3, Task 5), neutral rows kept so "no prefix" stays neutral (anchor rows, Task 5), emotion2vec+ accuracy with the CosyVoice 3 text-unrelated figures (Task 7), tag realisation by detector and by ear (Tasks 1, 7). Deviations from the spec, each from stage 1 evidence: no "continue the same adapter" (fresh LM-only LoRA from base); unlabelled replay comes from base-model outputs, not corpus or FLEURS clips; EmpathicInsight agreement dropped (calibration 2026-10-27: angry precision 0.40); gender in the prefix dropped (stage 2 speaks in clip B's voice).
- Engelsiz-specific sources used: subtitle sound cues (event clips, all usable ones cleaned in Task 3A), audio-description narration (Task 3B). Subtitle speaking-manner cues are too few (85) to train on; they stay a free check.
- Calls made by the planner, not the user: rank 32 and 2 epochs; anchor share 0.5; label quota 6,000 per style; the 0.8 bar for unconfirmed cries; the narrator lexicon, link window (−2 s … +20 s) and the 3,000-clip bar for a new style; `--labelled-ref-ratio 0.8`; Freya line ranges; the number thresholds in Task 7 other than the regression rule; the Sidon-check threshold (0.8×); the P1–P4 probe setups and English prompt wording.
- Known unknowns: whether VoxCPM2 follows a style prefix at all while cloning (Task 4 answers it); real-time factor for the anchor synthesis (Task 5 Step 5 measures it); whether `[iç çekme]` (138 usable clips) or `[öksürük]` (285) reach `--min-tag 150` after Task 3A (not trained here; they stay metadata); happy labels rest on 19 rated clips, so happy's numeric gate is lower and the ear decides.
