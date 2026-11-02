#!/usr/bin/env python
"""Rate the 300 hand-rated calibration clips with Qwen3-Omni (audio in, one emotion word out) and compare it with
emotion2vec+ on the same clips. Experiment for stage 2 Task 3B; nothing in the pipeline reads its output.

    .venv/bin/python scripts/omni_rater.py --clips data/calib_omni/calibrate_emotions --limit 5   # smoke test
    .venv/bin/python scripts/omni_rater.py --clips data/calib_omni/calibrate_emotions             # all 300, resumable

Confidence is the model's next-token probability of each class word (renormalised over the five words), so clips
can be ranked like emotion2vec+ scores instead of trusting a self-reported number.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttsdata.emotion import parse_ratings  # noqa: E402

WORDS = ("angry", "happy", "sad", "neutral", "other")
PROMPT = (
    "You hear a short clip of Turkish speech taken from a TV drama. Judge the speaker's emotional state from the "
    "VOICE ALONE: tone, pitch, loudness, tempo, voice quality. Do not use the meaning of the words. "
    "Pick exactly one of: angry (anger, irritation, contempt), happy (joy, amusement, warmth), sad (sadness, grief, "
    "disappointment), neutral (calm or matter-of-fact delivery, no clear emotion), other (fear, surprise, disgust, "
    "mixed, or more than one speaker). Answer with that one lowercase word only."
)


def load16k(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def load_model(model_id: str, quant: str):
    import torch
    from transformers import BitsAndBytesConfig, Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    kw = dict(dtype=torch.bfloat16, enable_audio_output=False)
    if quant == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["audio_tower", "visual", "lm_head", "talker", "code2wav", "gate"])
        kw["device_map"] = "cuda:0"
    else:  # bf16 split between GPU and CPU RAM
        kw["device_map"] = "auto"
        kw["max_memory"] = {0: "29GiB", "cpu": "52GiB"}
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_id, **kw)
    model.eval()
    proc = Qwen3OmniMoeProcessor.from_pretrained(model_id)
    return model, proc


def word_ids(tok) -> list[int]:
    ids = [tok.encode(w, add_special_tokens=False)[0] for w in WORDS]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"class words share a first token: {ids}")
    return ids


def rate(model, proc, ids: list[int], audio: np.ndarray) -> dict:
    import torch
    conv = [{"role": "user", "content": [{"type": "audio", "audio": "clip.wav"}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=text, audio=[audio], return_tensors="pt", padding=True)
    dev = next(model.thinker.parameters()).device
    inp = {k: (v.to(dev, dtype=torch.bfloat16) if v.is_floating_point() else v.to(dev)) for k, v in inp.items()}
    with torch.no_grad():
        logits = model.thinker(**inp).logits[0, -1].float()
    p_all = torch.softmax(logits, -1)
    sel = p_all[ids]
    probs = (sel / sel.sum()).tolist()
    return {"probs": dict(zip(WORDS, [round(x, 4) for x in probs])), "mass": round(float(sel.sum()), 4),
            "top_token": proc.tokenizer.decode([int(logits.argmax())])}


def precision_at(scores: dict[str, dict[str, float]], ratings: dict[str, str], cls: str, k: int) -> str:
    ranked = sorted((i for i in scores if i in ratings), key=lambda i: -scores[i].get(cls, 0.0))[:k]
    hit = sum(1 for i in ranked if ratings[i] == cls)
    return f"{hit}/{len(ranked)}"


def report(rows: list[dict], e2v: dict[str, dict[str, float]], ratings: dict[str, str], model_id: str) -> str:
    q = {r["id"]: r["probs"] for r in rows}
    ids = [i for i in q if i in e2v and i in ratings]
    q = {i: q[i] for i in ids}  # rank both models over the same clips
    e2v = {i: e2v[i] for i in ids}
    rated = {c: sum(1 for i in ids if ratings[i] == c) for c in ("angry", "happy", "sad", "neutral")}

    def acc(scores, classes):
        ok = sum(1 for i in ids if max(classes, key=lambda c: scores[i].get(c, 0.0)) == ratings[i])
        return f"{ok}/{len(ids)}"

    cls4 = ("angry", "happy", "sad", "neutral")
    lines = [f"# Qwen3-Omni against emotion2vec+ on the hand-rated clips", "",
             f"Model `{model_id}`, {len(ids)} clips with a rating and both scores. Rated: "
             + ", ".join(f"{c} {n}" for c, n in rated.items()) + ".", "",
             f"Top class over angry / happy / sad / neutral matches the rating: Qwen3-Omni {acc(q, cls4)}, "
             f"emotion2vec+ {acc(e2v, cls4)}.", "",
             "Precision of the k clips each model scores highest for a class (correct / k):", "",
             "| class | k | Qwen3-Omni | emotion2vec+ |", "|---|---|---|---|"]
    for c in ("angry", "sad", "happy"):
        for k in (10, 20, 30, 40):
            lines.append(f"| {c} | {k} | {precision_at(q, ratings, c, k)} | {precision_at(e2v, ratings, c, k)} |")
    m = [r["mass"] for r in rows]
    lines += ["", f"Probability mass on the five class words: median {np.median(m):.3f}, min {min(m):.3f}."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/calib_omni/calibrate_emotions")
    ap.add_argument("--ratings", default="calibration/emotion_ratings_2026-10-27.csv")
    ap.add_argument("--e2v", default="data/calib_omni/emotions_local.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--quant", choices=["nf4", "bf16"], default="bf16")
    ap.add_argument("--out", default="data/calib_omni/omni_rows.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    ratings = parse_ratings(Path(a.ratings).read_text(encoding="utf-8"))
    files = {p.name.split("_", 1)[1][:-4]: p for p in sorted(Path(a.clips).glob("*.wav"))}
    todo = [i for i in files if i in ratings][: a.limit or None]
    out = Path(a.out)
    done = {}
    if out.exists():
        done = {d["id"]: d for d in map(json.loads, out.read_text(encoding="utf-8").splitlines()) if d}
    model, proc = load_model(a.model, a.quant)
    ids = word_ids(proc.tokenizer)
    t0 = time.time()
    with out.open("a", encoding="utf-8") as f:
        for n, cid in enumerate([i for i in todo if i not in done], 1):
            r = {"id": cid, **rate(model, proc, ids, load16k(files[cid]))}
            done[cid] = r
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            if n <= 5 or n % 25 == 0:
                print(f"[{n}] {cid} {r['top_token']!r} {r['probs']} mass {r['mass']} rated {ratings[cid]} "
                      f"{time.time() - t0:.0f}s", flush=True)
    e2v = {}
    for l in Path(a.e2v).read_text(encoding="utf-8").splitlines():
        d = json.loads(l)
        if d.get("e2v"):
            e2v[d["id"]] = d["e2v"]
    text = report([done[i] for i in todo if i in done], e2v, ratings, a.model)
    Path(out.with_name("OMNI_RATER.md")).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
