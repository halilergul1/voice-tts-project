"""Score a run's (or any folder's) wavs for speaking style and sound tags: emotion2vec+ posteriors, style hit rate,
laugh realisation from the run's Whisper transcripts, voice similarity to a reference run.

  score_styles.py --run s2p_P2_angry_neutral --target angry --voice-ref-run s0_voxcpm2_contB_freya40
  score_styles.py --run s2p_P2_laugh --target neutral --tag laugh
  score_styles.py --run s2_sidoncheck_angry_sidon --wavs <folder of wavs> --target angry

Writes data/eval/<run>/styles.jsonl ({k, wav, e2v, hit, tag_realised, ref_cos}) and appends one row to
data/eval/style_summary.md. The run's scores.jsonl (eval_tts.py score) supplies the transcripts and read-aloud
words; without it, tag realisation and read-aloud stay empty. Runs in the venv with FunASR (the `emotions` extra).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttsdata.stylemetrics import style_hit, style_summary, tag_realised  # noqa: E402

EVAL_DIR = Path("data/eval")
SUMMARY = EVAL_DIR / "style_summary.md"
HEADER = ("| run | target | n | hit rate | target post | neutral rate | tag realised | read aloud | voice cos | date |\n"
          "|---|---|---|---|---|---|---|---|---|---|\n")


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def _wavs(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.wav"))


def _centre(paths: list[Path], enc, dev) -> np.ndarray:
    from ttsdata.speakers import embed_clips

    emb = embed_clips(paths, [sf.info(p).duration for p in paths], enc, dev, 16, desc="ecapa ref")
    c = emb.mean(axis=0)
    return c / (np.linalg.norm(c) + 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="output name under data/eval/; also the wav source unless --wavs")
    ap.add_argument("--wavs", default=None, help="score this folder of wavs instead of data/eval/<run>/wavs")
    ap.add_argument("--target", required=True, choices=["angry", "happy", "sad", "neutral"])
    ap.add_argument("--tag", default=None, choices=["laugh", "breath", "cry", "sigh"])
    ap.add_argument("--voice-ref-run", default=None, help="run whose mean ECAPA embedding is the voice to keep")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    from ttsdata.config import Config
    from ttsdata.emotion import E2V, _device

    run_dir = EVAL_DIR / a.run
    wav_dir = Path(a.wavs) if a.wavs else run_dir / "wavs"
    wavs = _wavs(wav_dir)
    if not wavs:
        raise SystemExit(f"no wavs in {wav_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    scores = {str(s["k"]): s for s in _load_jsonl(run_dir / "scores.jsonl")}
    key = lambda p: str(int(p.stem)) if p.stem.isdigit() else p.stem  # noqa: E731  (eval runs name wavs 0000.wav)

    cfg = Config()
    e2v = E2V(cfg.emotion_model, _device(a.device))
    ref_cos = {}
    if a.voice_ref_run:
        from ttsdata.speakers import embed_clips, load_encoder

        enc, dev = load_encoder(cfg)
        centre = _centre(_wavs(EVAL_DIR / a.voice_ref_run / "wavs"), enc, dev)
        emb = embed_clips(wavs, [sf.info(p).duration for p in wavs], enc, dev, 16, desc="ecapa")
        ref_cos = {p.stem: round(float(e @ centre), 4) for p, e in zip(wavs, emb)}

    rows = []
    for p in wavs:
        post = e2v.score(p)
        s = scores.get(key(p), {})
        rows.append({"k": key(p), "wav": str(p), "e2v": post, "hit": style_hit(post, a.target),
                     "tag_realised": tag_realised(s.get("asr", ""), a.tag) if a.tag and s else None,
                     "read_aloud": s.get("read_aloud"), "ref_cos": ref_cos.get(p.stem)})
    with (run_dir / "styles.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summ = style_summary(rows, a.target)
    realised = [r["tag_realised"] for r in rows if r["tag_realised"] is not None]
    aloud = [r for r in rows if r["read_aloud"] is not None]
    cos = [r["ref_cos"] for r in rows if r["ref_cos"] is not None]
    fmt = lambda x, d=3: "n/a" if x is None else f"{x:.{d}f}"  # noqa: E731
    line = (f"| {a.run} | {a.target} | {summ['n']} | {fmt(summ['hit_rate'], 2)} | {fmt(summ['target_post'])} | "
            f"{fmt(summ['neutral_rate'], 2)} | "
            f"{fmt(sum(realised) / len(realised) if realised else None, 2)} | "
            f"{sum(1 for r in aloud if r['read_aloud']) if aloud else 'n/a'} | "
            f"{fmt(float(np.mean(cos)) if cos else None)} | {date.today().isoformat()} |")
    if not SUMMARY.exists():
        SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY.write_text(HEADER, encoding="utf-8")
    with SUMMARY.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
