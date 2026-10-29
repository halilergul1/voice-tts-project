"""Second check after sidon_restore.py: did Sidon change words? (ttsdata.restoration.word_check)

Whisper (the project ASR, as in scripts/eval_tts.py) transcribes each restored clip of <root>/wavs_sidon; a clip whose
CER against the subtitle is clearly worse than the music-removed clip's (<root>/wavs_clean) is replaced by that clip.
Updates <root>/restoration.jsonl rows with cer_sidon / cer_clean and source "fallback_words".

  ~/tts/venv-voxcpm/bin/python scripts/sidon_wordcheck.py --root data/dataset [--shard K/N] ; ... --merge
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttsdata.restoration import FIRST_GATE_CER, MAX_EXTRA_CER, word_check  # noqa: E402
from ttsdata.separation import shard  # noqa: E402


def _load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def _transcriber():
    from ttsdata.asr import _words_from_ct2, resolve_backend
    from ttsdata.config import Config

    cfg = Config()
    backend, model = resolve_backend(cfg)
    print(f"ASR backend: {backend} / {model}", flush=True)
    if backend == "mlx":
        import mlx_whisper

        return lambda p: mlx_whisper.transcribe(str(p), path_or_hf_repo=model, language="tr", temperature=0.0,
                                                condition_on_previous_text=False, verbose=None)["text"]
    return lambda p: _words_from_ct2(str(p), model, cfg)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/dataset")
    ap.add_argument("--out", default="wavs_sidon")
    ap.add_argument("--ref", default="wavs_clean")
    ap.add_argument("--first-gate", type=float, default=FIRST_GATE_CER)
    ap.add_argument("--max-extra", type=float, default=MAX_EXTRA_CER)
    ap.add_argument("--shard", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    rfile = root / "restoration.jsonl"
    rows = {r["id"]: r for r in _load(rfile)}
    if a.merge:
        files = sorted(root.glob("wordcheck.shard*.jsonl"))
        for f in files:
            for u in _load(f):
                rows.setdefault(u["id"], {"id": u["id"]}).update(u)
        tmp = rfile.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows.values()), encoding="utf-8")
        tmp.replace(rfile)
        for f in files:
            f.unlink()
        n = sum(r.get("source") == "fallback_words" for r in rows.values())
        print(f"merged {len(files)} shard files into {rfile}: {n} clips fall back for changed words")
        return 0

    import jiwer

    from ttsdata.textnorm import for_cer

    meta = {r["id"]: r for r in _load(root / "metadata.jsonl")}
    done = {u["id"] for f in root.glob("wordcheck.shard*.jsonl") for u in _load(f)}
    todo = [i for i, r in rows.items() if r.get("source") == "sidon" and "cer_sidon" not in r and i not in done]
    if a.shard:
        k, n = (int(x) for x in a.shard.split("/"))
        todo = shard(todo, k, n)
        out_file = root / f"wordcheck.shard{k}of{n}.jsonl"
    else:
        out_file = root / "wordcheck.shard0of1.jsonl"
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(rows)} restored clips, {len(todo)} to check", flush=True)
    if not todo:
        return 0
    transcribe = _transcriber()

    def cer(path: Path, ref: str) -> float:
        hyp = for_cer(transcribe(path))
        return float(jiwer.cer(ref, hyp)) if hyp else 1.0

    n_fb = 0
    with out_file.open("a", encoding="utf-8") as fh:
        for i, cid in enumerate(todo, 1):
            ref = for_cer(meta[cid].get("normalized_text") or meta[cid]["text"])
            cs = round(cer(root / a.out / f"{cid}.wav", ref), 4)
            source, cc = word_check(cs, meta[cid].get("cer"), lambda: cer(root / a.ref / f"{cid}.wav", ref),
                                    a.first_gate, a.max_extra)
            if source == "fallback_words":
                shutil.copyfile(root / a.ref / f"{cid}.wav", root / a.out / f"{cid}.wav")
                n_fb += 1
            fh.write(json.dumps({"id": cid, "source": source, "cer_sidon": cs,
                                 "cer_clean": None if cc is None else round(cc, 4)}) + "\n")
            if i % 1000 == 0 or i == len(todo):
                fh.flush()
                print(f"[{i}/{len(todo)}] fallbacks {n_fb}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
