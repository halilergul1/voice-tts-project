# Calibration sets

Human ratings that fix thresholds in the pipeline. Small, hand-made, worth keeping under version control.

- `emotion_ratings_2026-10-27.csv`: 300 clips from the local 15 h build (60 per emotion2vec+ coarse class incl.
  unlabelled, spread over the series), rated by one native listener as angry / happy / sad / neutral / other.
  `predicted` and `conf` are emotion2vec+ large under the default cut (confidence 0.6, margin 0.2). Re-score with
  `ttsdata emotions --ratings calibration/emotion_ratings_2026-10-27.csv` (needs `emotions.jsonl` for the same clip ids);
  the report lands in `data/dataset/EMOTION_CALIBRATION.md`. Outcome: `--auto-conf angry=0.70,sad=0.85,happy=0.99`.
- `llm_gemini-3.5-flash-lite_2026-10-27.jsonl` / `LLM_RATER_gemini-3.5-flash-lite_2026-10-27.md`: the same 299 clips rated
  blind by Gemini 3.5 Flash-Lite (`scripts/llm_rater.py`, audio only, 25 clips per request): accuracy 0.50, precision
  angry 0.44 / sad 0.31 / happy 0.35, no useful confidence. `llm_gemini-3.5-flash_2026-10-27_partial.jsonl`: Gemini 3.5
  Flash on the 12 clips the free quota allowed. Neither is used.
