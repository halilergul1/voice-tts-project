# voice-data: Turkish TTS corpus from accessible (Engelsiz) TV broadcasts

Goal: a large, clean Turkish speech + text corpus for training / fine-tuning open-source TTS
models, built from the accessible ("Engelsiz") versions of Turkish TV series. Those broadcasts
ship human-made Turkish subtitles, which we align to the audio.

Sources

| source | where | how |
|---|---|---|
| Engelsiz TRT | YouTube channel, 1064 videos / ~1700 h listed in `meta_datasets/engelsiz_TRT_videos_metadata.csv` | `yt-dlp`: best audio + manual `tr` subtitles |
| Engelsiz Kanal D | `engelsiz.kanald.com.tr`, media on `cdn.duhnet.tv` | direct CDN fetch of the AAC track + sibling `.srt` |

## Current dataset (built 2026-10-25)

`data/dataset/`: **21662 utterances, 18.06 h**, 24 kHz mono, train 21229 / val 433.
Built from 19 episodes (40.4 h of broadcast): 9 single episodes across 9 Engelsiz TRT series + all 10
episodes of Eşref Rüya (Engelsiz Kanal D). Median clip 2.7 s (p5 1.3 s, p95 5.7 s). Mean subtitle-vs-Whisper
CER on the kept clips 0.03. Independent re-transcription of 100 random final clips: median CER 0, 94 within 0.2.

| series | episodes | utterances | hours |
|---|---|---|---|
| kanald: Eşref Rüya | 10 | 11181 | 8.68 |
| trt: Adını Sen Koy | 1 | 406 | 0.33 |
| trt: Filinta | 1 | 826 | 0.73 |
| trt: Hangimiz Sevmedik | 1 | 2147 | 1.87 |
| trt: Kalk Gidelim | 1 | 1589 | 1.34 |
| trt: Payitaht Abdülhamid | 1 | 1466 | 1.37 |
| trt: Seksenler | 1 | 1165 | 1.03 |
| trt: Sevda Kuşun Kanadında | 1 | 916 | 0.99 |
| trt: Yeşil Deniz | 1 | 1497 | 1.28 |
| trt: Yunus Emre | 1 | 469 | 0.45 |

Rejected candidates (30005 windows considered): high_cer 3523, multi_speaker 2723, unaligned 860, too_short 607, low_speech_ratio 370, no_speech 103, cps_too_low 70, credits 68, cps_too_high 6, too_long 5, text_too_short 4, bad_charset 4.
Roughly 45% of raw broadcast time ends up as usable speech; the rest is music, narration, overlapping
speakers, interjections too short to keep, and clips whose subtitle does not match the audio.

## Pipeline

```
download  ->  segment  ->  verify  ->  build  ->  speakers  ->  merge-clips    ->  soundtags        ->  extend-events
 raw/          segments/      asr.jsonl     dataset/    speaker_id    dataset_long/     soundtags.jsonl     dataset_events/
```

1. **download** (`ttsdata download-trt`, `ttsdata download-kanald`): audio + subtitles + `info.json` into `data/raw/<source>/<uid>/`. Videos without human-made Turkish subtitles are skipped (auto captions are never used).
2. **segment** (`ttsdata segment`): parse VTT/SRT; drop sound descriptions `(Jenerik müziği)`, lyrics (`♪` or quoted), cues with two speakers (dash dialogue, two labels, chorus labels); keep off-screen speaker labels like `(Usta)` as `speaker_hint`; join `...` continuations when the gap is short. Cut a 1 s padded window per cue, run Silero VAD to drop windows without speech, write 24 kHz mono WAVs.
3. **verify** (`ttsdata verify`): Whisper large-v3-turbo with word timestamps (MLX on Apple Silicon, faster-whisper/CUDA elsewhere). Clips are packed several at a time into one 30 s window separated by 1 s of silence (`pack_seconds`), transcribed once, and the words handed back to their clip by time; this fills the window Whisper pads to anyway, so it is ~3× faster than one call per clip and recognises short clips slightly better thanks to more acoustic context (`pack_seconds 0` restores one call per clip). Whisper words are aligned to the subtitle tokens with a fuzzy edit-distance; the clip is trimmed to exactly the aligned span (this removes the neighbouring speaker's tail and audio-description narration), and CER is computed on that span. Resumable.
4. **build** (`ttsdata build`): recompute edges from the word span, snap them with Silero VAD so leading non-speech is ≤ 0.15 s and trailing ≤ 0.25 s (Whisper dates a first word from the end of the preceding silence), filter (duration 1–15 s, CER ≤ 0.20, no-speech prob, chars/sec, charset), loudness-normalise to −20 LUFS, write the dataset.
5. **speakers** (`ttsdata speakers`): one ECAPA-TDNN speaker embedding per clip (`speechbrain/spkrec-ecapa-voxceleb`, CPU / Apple MPS / CUDA; cached per episode in `data/speakers/`, so re-runs only embed new episodes), agglomerative clustering (average linkage, cosine distance ≤ `speaker_threshold` 0.70) within each episode, then episode clusters of the same series merged by centroid distance (`speaker_merge_threshold` 0.55; two clusters of one episode never merge). Writes `speaker_id` (`esrefruya_spk03`, numbered by size within the series) and `speaker_cluster_size` to `metadata.jsonl`, and `speaker_id` as a 4th column of the csv files. Needs the `speakers` extra (`uv pip install --python .venv/bin/python -e ".[speakers]"`).
6. **merge-clips** (`ttsdata merge-clips`): joins consecutive subtitle lines of one voice into longer clips, because two thirds of the clips are under the 3 s VoxCPM wants. A link needs no dropped line in between, a pause ≤ 0.7 s, the same split, compatible speaker labels, and ECAPA cosine ≥ 0.4 to the previous line and to the chain centroid (≥ 0.5 across speaker clusters, ≥ 0.55 after a line ending in "?", where turn changes hide). The span is cut again from the raw episode audio, loudness-normalised and re-checked with Whisper (CER ≤ 0.10). Output `data/dataset_long/` in the dataset layout, each row listing its `members`; a manifest uses a merged clip or its members, never both. Wavs are cut atomically and re-cut when a rebuild moves a span (`cut_index.json`); Whisper verdicts are cached in `asr_cache.jsonl` per span, text and recogniser. `--limit-episodes N` writes `data/dataset_long_smoke/` instead. `listen_merges/index.md` holds the weakest voice links to check by ear.
7. **soundtags** (`ttsdata soundtags`): keeps the subtitle sound descriptions the segmenter drops (`(Gülme sesi)`, `(Hatice ağlama sesi)`, `(Ağlayarak) Neden?`, `(Gerilim müziği)`, `(***)`), classifies them (vocal event, speaking manner, music, sfx, background laugh, overlapping or foreign or unintelligible speech, censored word, TV / PA voice) and attaches them to clips of `data/dataset/` and `data/dataset_long/`: a speaking `style` (a merged clip only when every member has it), non-verbal `events` inside, before / after (within 1.5 s) or leading the clip's own line (`(Öksürme sesi) Ben...`), with episode times, and noise `flags`. Labels on a speech line are read as speaker names unless they clearly name a sound or a manner. Writes `soundtags.jsonl` per root, `sound_events.jsonl` (every description), `SOUNDTAGS.md` and `listen_tags/` (including raw cuts of event cues to check subtitle timing).
8. **extend-events** (`ttsdata extend-events`): for every kept clip with a subtitle laugh / cry / sigh / breath cue right before or after its line, re-cuts the clip from the raw audio to include the sound (cue ± 0.2 s, ≤ 15 s, never overlapping another line, never a sound the subtitle attributes to someone else), re-checks the words with Whisper (CER ≤ 0.10 and at most one stray token on the extended side) and tags the text: `[gülme] Hadi!`, `Neden gittin? [ağlama]`. Output `data/dataset_events/` in the dataset layout; each row names its `base` clip (a manifest uses one or the other) and stays `verified: false` until a sound-event detector confirms the sound. `listen_events/index.md` for checking by ear.

Dataset layout (`data/dataset/`): `wavs/*.wav`, `metadata.csv` (LJSpeech `id|text|normalized_text|speaker_id`), `train.csv`, `val.csv`, `metadata.jsonl` (rich per-utterance record), `rejected.jsonl` (why each candidate was dropped), `stats.json`, `README.md`.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[speakers]"   # speakers extra = speechbrain, for `ttsdata speakers`
```

Needs `ffmpeg` and `node` (yt-dlp's JavaScript runtime) on PATH. Whisper weights (~1.6 GB) download from Hugging Face on first `verify`.

## Tests

```bash
.venv/bin/python -m pytest tests
```

## Run

```bash
scripts/run_pipeline.sh            # segment + 3 parallel Whisper workers + build, for everything in data/raw
WORKERS=2 scripts/run_pipeline.sh  # fewer GPU workers
```

`verify` is resumable and lock-protected per episode, so it can be interrupted and re-run, and several
workers can share one queue (`scripts/verify_parallel.sh N`). Throughput on an M4 Max is roughly 1 hour
of broadcast audio verified per 4-5 minutes with 3 workers.

Add material:

```bash
# TRT: by id, or pick from the metadata csv
.venv/bin/ttsdata download-trt --ids GENdU9GxZ9g
.venv/bin/ttsdata download-trt --csv meta_datasets/engelsiz_TRT_videos_metadata.csv --series "Seksenler" --limit 20
# Kanal D: the CDN path pattern per series comes from the player page (see notebooks/kanald_scrape.ipynb)
.venv/bin/ttsdata download-kanald --slug esrefruya --name "Eşref Rüya" \
   --url-template "https://cdn.duhnet.tv/C13/2013kanald/dizi/esrefruya/er{n}.mp4" --episodes 1-10
.venv/bin/ttsdata segment && .venv/bin/ttsdata verify && .venv/bin/ttsdata build && .venv/bin/ttsdata speakers
```

Every threshold in `ttsdata/config.py` can be overridden on the command line, e.g. `ttsdata build --max-cer 0.1 --min-dur 2`.

Audit a built dataset (integrity, audio format, duration profile, and an independent Whisper
re-transcription of N random final clips):

```bash
.venv/bin/python scripts/audit_dataset.py --n 100
```

## Scaling up on a Linux GPU server

`verify` is the only expensive stage (~5.5 GPU-minutes per broadcast hour on an M4 Max with MLX; a
single modern NVIDIA card with faster-whisper is several times faster). The code picks its backend
automatically: MLX on Apple Silicon, faster-whisper (CTranslate2) elsewhere.

```bash
# on the server (needs ffmpeg, node, CUDA driver)
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[cuda,speakers]"
# optional explicit choice: --whisper-backend faster --whisper-device cuda --whisper-compute float16
WORKERS=2 scripts/run_pipeline.sh          # 2-4 workers per GPU, more if VRAM allows
```

Downloading from YouTube on a datacentre IP often triggers bot checks. Safest split: download on the
Mac (`download-trt`, fast and resumable), `rsync data/raw/` to the server, run `segment`/`verify`/`build`
there, and `rsync data/dataset/` back. `data/segments/` (windows) can be deleted after `build`.

Whole-catalogue run (downloads + processing loop + final build, resumable; re-run the same command after
any interruption and it continues):

```bash
# on the server, inside ~/voice-data
tmux new-session -d -s ttsdata 'N_DL=2 N_SEG=4 N_WORKERS=4 scripts/catalogue_driver.sh'
tail -f data/logs/pipeline.log                      # one line per Whisper pass: episodes, clips verified, disk
tail -f data/logs/download_0.log data/logs/download_1.log
ls data/raw/trt | wc -l; cat data/segments/*/*/asr.jsonl | wc -l   # episodes fetched, clips verified
nvidia-smi; df -h ~                                  # GPU busy? disk?
tmux attach -t ttsdata                               # Ctrl-b d to detach again
```

The driver runs three stages concurrently: `N_DL` downloaders (sharded over the csv), `N_SEG` segmenters
(lock-protected, CPU) and `N_WORKERS` Whisper workers (GPU); each stage drains when the previous one is done.
`SERIES="Seksenler|Payitaht" LIMIT=20 scripts/catalogue_driver.sh` restricts the run. Videos whose
YouTube captions are auto-generated only are skipped and logged (`[skip] <id>: no human-made Turkish subtitles`).
When `data/logs/pipeline_done` appears the dataset is in `data/dataset/`; copy it back with
`rsync -az ibrahim@astrobot.local:~/voice-data/data/dataset/ ./data/dataset_full/`.

Rough budget for the whole Engelsiz TRT catalogue (1,064 videos, ~1,700 h): ~70 GB raw audio, ~2 TB of
candidate windows if kept all at once (process in batches of ~100 episodes), and roughly a day of GPU time: an RTX 5090 verifies ~6 clips/s per worker with faster-whisper float16
(the M4 Max does ~2 clips/s in total).

Speaker-id quality on the 15 h local build (sweep against the subtitle speaker labels): threshold 0.70 /
merge 0.55 gives purity 0.91, completeness 0.72, 1,695 ids of which 266 have ≥ 10 clips and cover 86% of
all clips. Loosening the within-episode threshold raises completeness only by losing purity (0.75 → 0.81,
0.80 → 0.65), so the defaults favour clean clusters over few clusters; for training, keep ids with enough
clips (`speaker_cluster_size`) and treat the long tail as unlabelled. Reproduce with
`ttsdata speakers --sweep 0.70,0.75 --sweep-merge 0.55,0.75`.

## Known properties of the data

- Multi-speaker. `speaker_id` is unsupervised (clustered speaker embeddings, see below); `speaker_hint` is the subtitle's own label and is only set when it labelled an off-screen speaker (~6% of clips).
- Background music / effects are present under dialogue (TV mix). Clips where Whisper could not recover the subtitle text (CER > 0.2) are rejected, which removes most heavily masked speech.
- The accessible mix contains an audio-description narrator between dialogue lines. The alignment trim keeps only the subtitled span, so narration is excluded from clips; it is not part of the corpus.
- Text is the broadcaster's subtitle: casing and punctuation preserved; numbers expanded to words in `normalized_text`.

## Speaker ids

`ttsdata speakers` on the current dataset: **1695 ids** over 19,980 clips; 266 ids have ≥ 10 clips and hold 86% of the clips, the rest is a long tail of 1–2 clip ids (noisy or very short clips that the embedding does not place). Checked against the 1198 clips whose `speaker_hint` names a character (207 names after folding `X ses` / `X dış ses` into `X`): purity 0.91 (clips in an id that carry the id's majority name), completeness 0.72 (clips of a name that sit in the name's biggest id; off-screen / phone voice of the same character often lands in its own id). Threshold sweep (merge threshold 0.55):

| threshold | purity | completeness | ids | ids ≥ 10 clips | clips covered |
|---|---|---|---|---|---|
| 0.60 | 0.958 | 0.565 | 3656 | 330 | 0.79 |
| 0.65 | 0.937 | 0.640 | 2555 | 296 | 0.79 |
| **0.70** | **0.912** | **0.722** | **1695** | **266** | **0.86** |
| 0.75 | 0.836 | 0.761 | 1048 | 221 | 0.91 |
| 0.80 | 0.669 | 0.776 | 602 | 166 | 0.95 |

Per series: Eşref Rüya 582 ids (78 with ≥ 10 clips, purity 0.86), Hangimiz Sevmedik 230 (31), Kalk Gidelim 193 (23), Yeşil Deniz 153 (29), Seksenler 152 (19), Sevda Kuşun Kanadında 102 (26), Payitaht Abdülhamid 99 (25, purity 0.97), Filinta 92 (17), Adını Sen Koy 60 (9), Yunus Emre 32 (9). For speaker-conditioned training keep ids with `speaker_cluster_size` ≥ 10 (or ≥ 30 for the main cast); a lower `--speaker-threshold` gives purer but more fragmented ids. `ttsdata speakers --sweep 0.6,0.7,0.8` prints this table for the current data.

`notebooks/` holds the original exploration (channel metadata scrape, Kanal D Selenium scraper). The Selenium notebook is still the way to discover a new series' CDN path pattern.
