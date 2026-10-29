"""Restore training clips with Sidon (sarulab-speech/sidon-v0.1) from their context cuts, with an ending check.

Input: <root>/<in>/<id>.wav = the clip with context, music removed (scripts/cut_context.py, then separate_vocals.py
--wavs wavs_ctx --out wavs_ctx_clean), and <root>/ctx.jsonl (lead, dur). Per clip: faint-noise padding, Sidon, trim
back to the clip, loudness-normalise. When the last word comes out more than --max-end-drop dB quieter than in the
music-removed clip (<root>/<ref>/<id>.wav), that clip is kept instead (fallback). Writes <root>/<out>/<id>.wav and
<root>/restoration.jsonl rows {id, source: sidon|fallback, end_drop_db}. Resumable; --shard K/N as in
separate_vocals.py (shard files restoration.shard<K>of<N>.jsonl, folded in by --merge).

Run in the venv with torch + transformers (~/tts/venv-voxcpm):
  sidon_restore.py --root data/dataset [--shard 0/2] ;  sidon_restore.py --root data/dataset --merge
Restoration follows the official demo (huggingface.co/spaces/sarulab-speech/sidon_demo_beta).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttsdata.restoration import MAX_END_DROP_DB, ending_drop_db, noise_pad, trim_span  # noqa: E402
from ttsdata.separation import loudness_normalize, shard  # noqa: E402

REPO = "sarulab-speech/sidon-v0.1"
SR_IN, SR_OUT, SR_SAVE = 16_000, 48_000, 24_000
PAD_S = 1.5


def _load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


class Sidon:
    def __init__(self, device: str = "cuda"):
        import torch
        import transformers
        from huggingface_hub import hf_hub_download

        self.torch, self.device = torch, device
        self.fe = torch.jit.load(hf_hub_download(REPO, f"feature_extractor_{device}.pt"), map_location=device).to(device)
        self.dec = torch.jit.load(hf_hub_download(REPO, f"decoder_{device}.pt"), map_location=device).to(device)
        self.pre = transformers.SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        self.rng = np.random.default_rng(0)

    def __call__(self, y: np.ndarray, sr: int, lead: float, dur: float) -> np.ndarray:
        """Restore a context cut, return the clip span at SR_SAVE."""
        import torchaudio

        torch = self.torch
        w = torch.from_numpy(0.9 * y / (np.abs(y).max() + 1e-9)).view(1, -1)
        w = torchaudio.functional.highpass_biquad(w, sr, 50)
        x = torchaudio.functional.resample(w, sr, SR_IN).view(-1).numpy()
        pad = int(PAD_S * SR_IN)
        x = torch.from_numpy(noise_pad(x, pad, self.rng))
        with torch.inference_mode():
            inp = self.pre(torch.nn.functional.pad(x, (160, 160)), return_tensors="pt", sampling_rate=SR_IN)
            feat = self.fe(inp["input_features"].to(self.device))["last_hidden_state"]
            out = self.dec(feat.transpose(1, 2)).view(-1).float().cpu()
        out = trim_span(out.numpy(), SR_OUT, PAD_S + lead, dur)
        return torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(out)).view(1, -1), SR_OUT,
                                              SR_SAVE).view(-1).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/dataset")
    ap.add_argument("--in", dest="inp", default="wavs_ctx_clean")
    ap.add_argument("--ref", default="wavs_clean")
    ap.add_argument("--out", default="wavs_sidon")
    ap.add_argument("--max-end-drop", type=float, default=MAX_END_DROP_DB)
    ap.add_argument("--target-lufs", type=float, default=-20.0)
    ap.add_argument("--peak", type=float, default=0.99)
    ap.add_argument("--shard", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    rfile = root / "restoration.jsonl"
    if a.merge:
        files = sorted(root.glob("restoration.shard*.jsonl"))
        rows = {r["id"]: r for r in _load(rfile)}
        rows.update({r["id"]: r for f in files for r in _load(f)})
        tmp = rfile.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows.values()), encoding="utf-8")
        tmp.replace(rfile)
        for f in files:
            f.unlink()
        n_fb = sum(r["source"] == "fallback" for r in rows.values())
        print(f"merged {len(files)} shard files: {len(rows)} clips in {rfile}, {n_fb} fallbacks")
        return 0
    inp, ref, out = root / a.inp, root / a.ref, root / a.out
    out.mkdir(parents=True, exist_ok=True)
    ctx = [c for c in _load(root / "ctx.jsonl") if (inp / f"{c['id']}.wav").exists()]
    done = {p.stem for p in out.glob("*.wav")}
    todo = [c for c in {c["id"]: c for c in ctx}.values() if c["id"] not in done]
    if a.shard:
        k, n = (int(x) for x in a.shard.split("/"))
        ids = set(shard([c["id"] for c in todo], k, n))
        todo = [c for c in todo if c["id"] in ids]
        rfile_out = root / f"restoration.shard{k}of{n}.jsonl"
    else:
        rfile_out = root / "restoration.shard0of1.jsonl"
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(ctx)} context cuts, {len(done)} restored, {len(todo)} to do", flush=True)
    if not todo:
        return 0
    sidon = Sidon()
    n_fb = 0
    with rfile_out.open("a", encoding="utf-8") as fh:
        for i, c in enumerate(todo, 1):
            y, sr = sf.read(inp / f"{c['id']}.wav", dtype="float32")
            y = y if y.ndim == 1 else y.mean(1)
            r, rsr = sf.read(ref / f"{c['id']}.wav", dtype="float32")
            r = r if r.ndim == 1 else r.mean(1)
            restored = sidon(y, sr, c["lead"], c["dur"])
            drop = round(ending_drop_db(restored, SR_SAVE, r, rsr), 2)
            if drop < -a.max_end_drop:
                shutil.copyfile(ref / f"{c['id']}.wav", out / f"{c['id']}.wav")
                source = "fallback"
                n_fb += 1
            else:
                sf.write(out / f"{c['id']}.wav", loudness_normalize(restored, SR_SAVE, a.target_lufs, a.peak), SR_SAVE,
                         subtype="PCM_16")
                source = "sidon"
            fh.write(json.dumps({"id": c["id"], "source": source, "end_drop_db": drop}) + "\n")
            if i % 500 == 0 or i == len(todo):
                fh.flush()
                print(f"[{i}/{len(todo)}] fallbacks {n_fb}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
