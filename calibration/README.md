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
- `omni_qwen3-omni_2026-11-02.jsonl` / `OMNI_RATER_qwen3-omni_2026-11-02.md`: the same clips scored by Qwen3-Omni-30B-A3B-Instruct
  (`scripts/omni_rater.py`, bf16 split GPU/CPU, class-word probabilities instead of a self-reported confidence). Top class
  matches the rating 175/268 against emotion2vec+ 172/268, but its most-confident clips are less often right in every class
  (angry 25/30 vs 30/30, sad 20/30 vs 23/30, happy 11/30 vs 16/30). Not used. The clips were sampled 60 per emotion2vec+
  class, which may flatter emotion2vec+ somewhat.
