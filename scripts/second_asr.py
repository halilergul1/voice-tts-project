"""Second speech recogniser for the Task 3D text vote (another model family than Whisper, so it does not repeat
Whisper's mistakes).

  second_asr.py bench --models seamless,mms --n 300          # pick a model: accuracy and how the vote would come out
  second_asr.py run --model seamless --ids ids.txt --out data/dataset/asr2.jsonl --shard 0/2

bench: "trusted" clips (Whisper matched the subtitle exactly, so the subtitle is what is said) measure the model's
accuracy on TV speech; "rejected" clips (high_cer, >= 1.5 s, the subtitle's own time window from data/segments) show
how often it backs the subtitle, backs Whisper, or neither, on the span Whisper aligned (the clip the dataset would
have cut; Whisper's text of that span). Writes data/eval/asr2_bench.jsonl and .md.
run: transcribes the clips listed in --ids (rejected ids resolve to their segment wav, others to data/dataset/wavs),
appends {id, model, text} to --out, skips ids already there.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttsdata.textvote import _cer, vote  # noqa: E402

MODELS = {"seamless": "facebook/seamless-m4t-v2-large", "mms": "facebook/mms-1b-all"}


def load_16k(path: Path, t0: float | None = None, t1: float | None = None) -> np.ndarray:
    import torch
    import torchaudio.functional as F

    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if t0 is not None:
        wav = wav[int(t0 * sr):int(t1 * sr)]
    if sr != 16000:
        wav = F.resample(torch.from_numpy(wav), sr, 16000).numpy()
    return wav


class Recogniser:
    def __init__(self, name: str, device: str = "cuda"):
        import torch
        from transformers import AutoProcessor

        self.name, self.device = name, device
        if name == "seamless":
            from transformers import SeamlessM4Tv2ForSpeechToText

            self.proc = AutoProcessor.from_pretrained(MODELS[name])
            self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained(MODELS[name], torch_dtype=torch.float16).to(device)
        elif name == "mms":
            from transformers import Wav2Vec2ForCTC

            self.proc = AutoProcessor.from_pretrained(MODELS[name], target_lang="tur")
            self.model = Wav2Vec2ForCTC.from_pretrained(MODELS[name], target_lang="tur", ignore_mismatched_sizes=True,
                                                        torch_dtype=torch.float16).to(device)
        else:
            raise ValueError(f"unknown model {name}; known: {', '.join(MODELS)}")
        self.model.eval()

    def __call__(self, wavs: list[np.ndarray]) -> list[str]:
        import torch

        with torch.inference_mode():
            if self.name == "seamless":
                x = self.proc(audio=wavs, sampling_rate=16000, return_tensors="pt", padding=True).to(self.device)
                x["input_features"] = x["input_features"].half()
                out = self.model.generate(**x, tgt_lang="tur")
                return [t.strip() for t in self.proc.batch_decode(out, skip_special_tokens=True)]
            x = self.proc(wavs, sampling_rate=16000, return_tensors="pt", padding=True).to(self.device)
            logits = self.model(x["input_values"].half(), attention_mask=x.get("attention_mask")).logits
            return [t.strip() for t in self.proc.batch_decode(logits.argmax(-1))]


def segment_wav(clip_id: str) -> Path | None:
    uid = clip_id.rsplit("_", 1)[0]
    for src in ("trt", "kanald"):
        p = Path("data/segments") / src / uid / f"{clip_id}.wav"
        if p.exists():
            return p
    return None


_ASR_CACHE: dict[str, dict] = {}


def aligned_span(clip_id: str) -> dict | None:
    """The rejected line as the dataset would have cut it: {wav, t0, t1, whisper} from the episode's asr.jsonl
    (trimmed to the span Whisper aligned to the subtitle; whisper = Whisper's text of that span)."""
    w = segment_wav(clip_id)
    if not w:
        return None
    uid = w.parent.name
    if uid not in _ASR_CACHE:
        _ASR_CACHE[uid] = {r["id"]: r for r in map(json.loads, (w.parent / "asr.jsonl").read_text(encoding="utf-8")
                                                    .splitlines()) if r}
    a = _ASR_CACHE[uid].get(clip_id) or {}
    if a.get("trim_start") is None or a.get("trim_end", 0) - a["trim_start"] < 1.0:
        return None
    return {"wav": str(w), "t0": a["trim_start"], "t1": a["trim_end"], "whisper": a.get("span_text") or ""}


def bench_sets(n: int, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    trusted = []
    for l in open("data/dataset/metadata.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r.get("cer") == 0.0 and 3.0 <= r["duration"] <= 12.0:
            trusted.append({"id": r["id"], "wav": str(Path("data/dataset") / r["audio_file"]), "subtitle": r["text"],
                            "whisper": r.get("asr_text") or ""})
    rejected = []
    for l in open("data/dataset/rejected.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["reason"] == "high_cer" and (r.get("duration") or 0) >= 1.5:
            rejected.append({"id": r["id"], "subtitle": r["text"], "whisper_cer": r.get("cer")})
    trusted = rng.sample(trusted, n)
    picked = []
    for r in rng.sample(rejected, min(len(rejected), 3 * n)):
        span = aligned_span(r["id"])
        if span and span["t1"] - span["t0"] >= 1.5:
            picked.append({**r, **span})
        if len(picked) == n:
            break
    return trusted, picked


def transcribe(rec: Recogniser, rows: list[dict], batch: int) -> float:
    t0 = time.time()
    for b in range(0, len(rows), batch):
        part = rows[b:b + batch]
        for r, t in zip(part, rec([load_16k(Path(r["wav"]), r.get("t0"), r.get("t1")) for r in part])):
            r[rec.name] = t
    return time.time() - t0


def cmd_bench(a) -> None:
    trusted, rejected = bench_sets(a.n, a.seed)
    print(f"bench: {len(trusted)} trusted, {len(rejected)} rejected clips")
    lines = ["# Second recogniser bench (Task 3D)", "",
             f"{len(trusted)} trusted clips (Whisper = subtitle, 3-12 s), {len(rejected)} high_cer rejects (>= 1.5 s, "
             "subtitle time window). Agreement = CER <= 0.05 after for_cer.", "",
             "| model | trusted CER (mean) | trusted CER <= 0.05 | rejected: backs subtitle | backs Whisper | neither "
             "| clips/s |", "|---|---|---|---|---|---|---|"]
    for name in a.models.split(","):
        rec = Recogniser(name)
        secs = transcribe(rec, trusted + rejected, a.batch)
        cers = [_cer(r["subtitle"], r[name]) for r in trusted]
        votes = [vote(r["subtitle"], r["whisper"], r[name]) for r in rejected]
        for r, v in zip(rejected, votes):
            r[f"vote_{name}"] = v
        nt, nr = len(trusted), len(rejected)
        lines.append(f"| {name} | {np.mean(cers):.3f} | {sum(c <= 0.05 for c in cers) / nt:.0%} | "
                     f"{votes.count('subtitle') / nr:.0%} | {votes.count('spoken') / nr:.0%} | "
                     f"{votes.count('drop') / nr:.0%} | {(nt + nr) / secs:.1f} |")
        print(lines[-1], flush=True)
        del rec
        import torch
        torch.cuda.empty_cache()
    out = Path("data/eval/asr2_bench")
    with out.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
        for r in [{**r, "set": "trusted"} for r in trusted] + [{**r, "set": "rejected"} for r in rejected]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def cmd_run(a) -> None:
    ids = [l.strip() for l in Path(a.ids).read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.shard:
        k, n = (int(x) for x in a.shard.split("/"))
        ids = ids[k::n]
    out = Path(a.out)
    done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()} if out.exists() else set()
    todo = []
    for i in ids:
        if i in done:
            continue
        if a.segments:
            span = aligned_span(i)
            if span:
                todo.append({"id": i, **span})
        elif (Path("data/dataset/wavs") / f"{i}.wav").exists():
            todo.append({"id": i, "wav": str(Path("data/dataset/wavs") / f"{i}.wav")})
    print(f"{len(ids)} ids, {len(todo)} to transcribe ({len(done)} already in {out})", flush=True)
    rec = Recogniser(a.model)
    with out.open("a", encoding="utf-8") as f:
        for b in range(0, len(todo), a.batch):
            part = todo[b:b + a.batch]
            transcribe(rec, part, a.batch)
            for r in part:
                f.write(json.dumps({"id": r["id"], "model": MODELS[a.model], "text": r[a.model],
                                    **({"t0": r["t0"], "t1": r["t1"], "whisper": r["whisper"]} if "t0" in r else {})},
                                   ensure_ascii=False) + "\n")
            f.flush()
            if b // a.batch % 50 == 0:
                print(f"{b + len(part)} / {len(todo)}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--models", default="seamless,mms")
    b.add_argument("--n", type=int, default=300)
    b.add_argument("--seed", type=int, default=1)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True, choices=list(MODELS))
    r.add_argument("--ids", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--segments", action="store_true", help="ids are rejected lines: read their segment wavs")
    r.add_argument("--shard", default="")
    for p in (b, r):
        p.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    {"bench": cmd_bench, "run": cmd_run}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
