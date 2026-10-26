"""Synthesise a sentence list with a TTS system and score it: Whisper WER/CER, UTMOSv2, voice consistency, RTF.

  eval_tts.py fetch                                    # Freya-TR-Eval sentences -> data/eval/freya_tr_eval.txt
  eval_tts.py synth --system voxcpm2 --run s0_voxcpm2 --sentences data/eval/freya_tr_eval.txt [--limit 20]
  eval_tts.py synth --system voxcpm2-lora --lora ~/tts/checkpoints/stage1/step_0001000 --run s1_step1000 ...
  eval_tts.py synth --system trendyol --run s0_trendyol ...
  eval_tts.py score --run s0_voxcpm2 --sentences data/eval/freya_tr_eval.txt [--utmos 0] [--spk 0]
  eval_tts.py all   ... (synth then score, same flags)

Outputs under data/eval/<run>/: wavs/NNNN.wav, synth.jsonl, scores.jsonl; one summary row appended to
data/eval/summary.md. Runs inside the venv that has voxcpm (+ the Whisper backend, utmosv2, speechbrain for scoring).
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttsdata.evalmetrics import lora_config_from_checkpoint, summarize  # noqa: E402

EVAL_DIR = Path("data/eval")
DEFAULT_PROMPT = "(Otuzlu yaşlarında bir erkek, sakin ve doğal bir ses)"
TEXT_KEYS = ("text", "sentence", "transcript", "transcription", "normalized_text")


def _device(choice: str) -> str:
    if choice != "auto":
        return choice
    import torch

    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def cmd_fetch(a) -> None:
    from huggingface_hub import snapshot_download

    repo = Path(snapshot_download(a.repo, repo_type="dataset"))
    texts: list[str] = []
    for p in sorted(repo.rglob("*")):
        if p.suffix == ".txt":
            texts += [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        elif p.suffix in (".jsonl", ".json"):
            rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.suffix == ".jsonl" \
                else json.loads(p.read_text(encoding="utf-8"))
            for r in rows if isinstance(rows, list) else []:
                for k in TEXT_KEYS:
                    if isinstance(r, dict) and r.get(k):
                        texts.append(str(r[k]).strip())
                        break
        elif p.suffix in (".csv", ".parquet"):
            import pandas as pd

            df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            col = next((k for k in TEXT_KEYS if k in df.columns), None)
            if col:
                texts += [str(t).strip() for t in df[col].tolist() if str(t).strip()]
    seen, out = set(), []
    for t in texts:
        if t not in seen:
            seen.add(t)
            out.append(t)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    dst = EVAL_DIR / a.out
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{len(out)} sentences -> {dst} (from {repo})")


def _load_voxcpm(system: str, model_dir: str | None, lora: str | None, device: str):
    from voxcpm import VoxCPM

    sig = inspect.signature(VoxCPM.from_pretrained).parameters
    kw = {}
    if "load_denoiser" in sig:
        kw["load_denoiser"] = False
    if "device" in sig:
        kw["device"] = device
    if "optimize" in sig:
        kw["optimize"] = device == "cuda"  # torch.compile path; keep off on MPS / CPU
    if system == "trendyol":
        model_id = model_dir or "Trendyol/Trendyol-TTS"
    else:
        model_id = model_dir or "openbmb/VoxCPM2"
        if system == "voxcpm2-lora":
            if not lora:
                raise SystemExit("--lora <step dir> is required for voxcpm2-lora")
            from voxcpm.model.voxcpm import LoRAConfig

            cfg = lora_config_from_checkpoint(Path(lora), inspect.signature(LoRAConfig).parameters)
            if cfg is None:
                raise SystemExit(f"{lora}/lora_config.json missing: inference LoRA settings must match training")
            kw["lora_config"] = LoRAConfig(**cfg)  # without it the library assumes r=8 / alpha=16
            kw["lora_weights_path"] = lora
    if "hf_model_id" in sig:
        return VoxCPM.from_pretrained(hf_model_id=model_id, **kw)
    return VoxCPM.from_pretrained(model_id, **kw)


def cmd_synth(a) -> None:
    sentences = [l.strip() for l in Path(a.sentences).read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.limit:
        sentences = sentences[: a.limit]
    run_dir = EVAL_DIR / a.run
    wav_dir = run_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    device = _device(a.device)
    model = _load_voxcpm(a.system, a.model_dir, a.lora, device)
    sr = model.tts_model.sample_rate
    inner = getattr(model, "_generate", model.generate)  # generate() is a *args/**kwargs wrapper around _generate()
    gen_sig = inspect.signature(inner).parameters
    takes_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in gen_sig.values())
    prompt = a.prompt if a.prompt is not None else (DEFAULT_PROMPT if a.system != "trendyol" else "")
    synth_file = run_dir / "synth.jsonl"
    rows_by_k: dict[int, dict] = {}
    if synth_file.exists() and not a.force:
        for l in synth_file.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                rows_by_k[r["k"]] = r

    def write_synth():
        with synth_file.open("w", encoding="utf-8") as f:
            for kk in sorted(rows_by_k):
                f.write(json.dumps(rows_by_k[kk], ensure_ascii=False) + "\n")

    if True:
        for k, text in enumerate(sentences):
            out = wav_dir / f"{k:04d}.wav"
            if out.exists() and not a.force:
                continue  # its synth.jsonl row (if any) is kept from the previous run
            full = f"{prompt} {text}".strip() if prompt else text
            kw = {"text": full, "cfg_value": a.cfg, "inference_timesteps": a.steps}
            for key, val in (("normalize", a.normalize), ("denoise", False)):
                if key in gen_sig or takes_kwargs:
                    kw[key] = val
            import torch

            torch.manual_seed(a.seed + k)
            t0 = time.time()
            wav = model.generate(**kw)
            wav = np.asarray(wav, dtype=np.float32).squeeze()
            secs = time.time() - t0
            sf.write(out, wav, sr, subtype="PCM_16")
            row = {"k": k, "text": text, "input": full, "audio_sec": round(len(wav) / sr, 3), "gen_sec": round(secs, 3),
                   "rtf": round(secs / max(len(wav) / sr, 1e-6), 3)}
            rows_by_k[k] = row
            write_synth()
            print(f"[{k + 1}/{len(sentences)}] {row['audio_sec']} s audio in {row['gen_sec']} s (rtf {row['rtf']})")
    (run_dir / "run.json").write_text(json.dumps({"system": a.system, "lora": a.lora, "model_dir": a.model_dir,
                                                  "prompt": prompt, "cfg": a.cfg, "steps": a.steps, "seed": a.seed,
                                                  "device": device, "sample_rate": sr, "sentences": a.sentences,
                                                  "n": len(sentences), "date": date.today().isoformat()},
                                                 ensure_ascii=False, indent=2), encoding="utf-8")


def _transcribe_fn(cfg):
    from ttsdata.asr import resolve_backend

    backend, model = resolve_backend(cfg)
    print(f"ASR backend: {backend} / {model}")
    if backend == "mlx":
        import mlx_whisper

        return lambda p: mlx_whisper.transcribe(str(p), path_or_hf_repo=model, language="tr", temperature=0.0,
                                                condition_on_previous_text=False, verbose=None)["text"]
    from ttsdata.asr import _words_from_ct2

    return lambda p: _words_from_ct2(str(p), model, cfg)[0]


def cmd_score(a) -> None:
    import jiwer

    from ttsdata.config import Config
    from ttsdata.textnorm import for_cer

    run_dir = EVAL_DIR / a.run
    wav_dir = run_dir / "wavs"
    sentences = [l.strip() for l in Path(a.sentences).read_text(encoding="utf-8").splitlines() if l.strip()]
    synth = {}
    if (run_dir / "synth.jsonl").exists():
        for l in (run_dir / "synth.jsonl").read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                synth[r["k"]] = r
    run_meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8")) if (run_dir / "run.json").exists() else {}
    if run_meta.get("sentences") and Path(run_meta["sentences"]).name != Path(a.sentences).name:
        print(f"[warn] run was synthesised from {run_meta['sentences']}, scoring against {a.sentences}")
    n_use = min(len(sentences), a.limit) if a.limit else len(sentences)
    wavs = [p for p in sorted(wav_dir.glob("*.wav")) if p.stem.isdigit() and int(p.stem) < n_use]
    if not wavs:
        raise SystemExit(f"no wavs in {wav_dir}: run synth first")
    cfg = Config()
    device = _device(a.device)
    transcribe = _transcribe_fn(cfg)
    scores = []
    for p in wavs:
        k = int(p.stem)
        ref, hyp = for_cer(sentences[k]), for_cer(transcribe(p))
        scores.append({"k": k, "text": sentences[k], "asr": hyp,
                       "wer": jiwer.wer(ref, hyp) if hyp else 1.0, "cer": jiwer.cer(ref, hyp) if hyp else 1.0,
                       "utmos": None, "spk_cos": None, "rtf": synth.get(k, {}).get("rtf")})
    by_k = {s["k"]: s for s in scores}
    if a.utmos:
        import utmosv2

        m = utmosv2.create_model(pretrained=True, device=device if device != "mps" else "cpu")
        for r in m.predict(input_dir=str(wav_dir), device=device if device != "mps" else "cpu"):
            k = int(Path(r["file_path"]).stem)
            if k in by_k:  # predict() scores the whole folder, also wavs beyond --limit / the sentence count
                by_k[k]["utmos"] = round(float(r["predicted_mos"]), 3)
    if a.spk:
        from ttsdata.speakers import embed_clips, load_encoder

        enc, dev = load_encoder(cfg)
        emb = embed_clips(wavs, [sf.info(p).duration for p in wavs], enc, dev, 16, desc="ecapa")
        centre = emb.mean(axis=0)
        centre /= np.linalg.norm(centre) + 1e-9
        for p, e in zip(wavs, emb):
            by_k[int(p.stem)]["spk_cos"] = round(float(e @ centre), 4)
    with (run_dir / "scores.jsonl").open("w", encoding="utf-8") as f:
        for s in scores:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    summ = summarize(scores)
    fmt = lambda x, d=3: "n/a" if x is None else f"{x:.{d}f}"  # noqa: E731
    row = (f"| {a.run} | {summ['n']} | {fmt(summ['wer'])} | {fmt(summ['cer'])} | {fmt(summ['utmos_mean'], 2)} | "
           f"{fmt(summ['utmos_p10'], 2)} | {fmt(summ['spk_consistency'])} | {fmt(summ['rtf'], 2)} | {date.today().isoformat()} |")
    summary = EVAL_DIR / "summary.md"
    if not summary.exists():
        summary.write_text("| run | n | WER | CER | UTMOS mean | UTMOS p10 | voice consistency | RTF | date |\n"
                           "|---|---|---|---|---|---|---|---|---|\n", encoding="utf-8")
    with summary.open("a", encoding="utf-8") as f:
        f.write(row + "\n")
    print(row)
    worst = sorted(scores, key=lambda s: -s["cer"])[:3]
    for s in worst:
        print(f"  CER {s['cer']:.2f}  TEXT: {s['text']}\n            ASR: {s['asr']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--repo", default="freyavoice/freya-tr-eval")
    f.add_argument("--out", default="freya_tr_eval.txt")
    f.set_defaults(func=cmd_fetch)
    for name in ("synth", "score", "all"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        s.add_argument("--sentences", required=True)
        s.add_argument("--device", default="auto")
        s.add_argument("--limit", type=int, default=0)
        if name in ("synth", "all"):
            s.add_argument("--system", choices=["voxcpm2", "voxcpm2-lora", "trendyol"], required=True)
            s.add_argument("--lora", default=None)
            s.add_argument("--model-dir", default=None, help="local model folder instead of the HF id")
            s.add_argument("--prompt", default=None, help="voice-design prefix; '' disables; default: Turkish male prompt")
            s.add_argument("--cfg", type=float, default=2.0)
            s.add_argument("--steps", type=int, default=16)
            s.add_argument("--seed", type=int, default=42)
            s.add_argument("--normalize", action="store_true", help="let VoxCPM normalise text (default: our text is already normalised)")
            s.add_argument("--force", action="store_true")
        if name in ("score", "all"):
            s.add_argument("--utmos", type=int, default=1)
            s.add_argument("--spk", type=int, default=1)
        s.set_defaults(func={"synth": cmd_synth, "score": cmd_score, "all": lambda a: (cmd_synth(a), cmd_score(a))}[name])
    a = ap.parse_args()
    a.func(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
