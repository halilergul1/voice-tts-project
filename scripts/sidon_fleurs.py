"""Sidon restoration of the FLEURS-tr replay clips (data/fleurs_tr: 24 kHz wav + .txt) for a clean-only build.

FLEURS clips are whole read sentences with silence at both ends, so no context cut is needed: Sidon runs on the clip
itself with faint-noise padding (scripts/sidon_restore.py) and the result is loudness-normalised. A clip is kept only
when its last word is not more than --max-end-drop dB fainter than in the original and Whisper finds no changed words
(ttsdata.restoration.word_check, the original recording standing in for the music-removed clip). Kept clips go to
--out as wav + txt; the others are left out, not copied, so the folder holds restored audio only.
Writes <out>/restoration.jsonl rows {id, source: sidon|fallback|fallback_words|unreadable, end_drop_db, cer_sidon,
cer_orig}. Resumable.

  ~/tts/venv-voxcpm/bin/python scripts/sidon_fleurs.py --in data/fleurs_tr --out data/fleurs_tr_sidon [--limit 20]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sidon_restore import SR_SAVE, Sidon  # noqa: E402
from sidon_wordcheck import _transcriber  # noqa: E402
from ttsdata.manifest import replay_text  # noqa: E402
from ttsdata.restoration import FIRST_GATE_CER, MAX_END_DROP_DB, MAX_EXTRA_CER, ending_drop_db, word_check  # noqa: E402
from ttsdata.separation import loudness_normalize  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/fleurs_tr")
    ap.add_argument("--out", default="data/fleurs_tr_sidon")
    ap.add_argument("--max-end-drop", type=float, default=MAX_END_DROP_DB)
    ap.add_argument("--first-gate", type=float, default=FIRST_GATE_CER)
    ap.add_argument("--max-extra", type=float, default=MAX_EXTRA_CER)
    ap.add_argument("--target-lufs", type=float, default=-20.0)
    ap.add_argument("--peak", type=float, default=0.99)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    inp, out = Path(a.inp), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rfile = out / "restoration.jsonl"
    done = {json.loads(l)["id"] for l in rfile.read_text(encoding="utf-8").splitlines() if l.strip()} if rfile.exists() else set()
    todo = [p for p in sorted(inp.glob("*.wav")) if p.stem not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(done)} clips done, {len(todo)} to do", flush=True)
    if not todo:
        return 0

    import jiwer

    from ttsdata.textnorm import for_cer

    sidon, transcribe = Sidon(), _transcriber()

    def cer(path: Path, ref: str) -> float:
        hyp = for_cer(transcribe(path))
        return float(jiwer.cer(ref, hyp)) if hyp else 1.0

    n = {"sidon": 0, "fallback": 0, "fallback_words": 0, "unreadable": 0}
    with rfile.open("a", encoding="utf-8") as fh:
        for i, p in enumerate(todo, 1):
            txt = p.with_suffix(".txt")
            text = replay_text(txt.read_text(encoding="utf-8")) if txt.exists() else None
            row = {"id": p.stem}
            if text is None:  # the builder skips these anyway (digits, foreign words, markup)
                row["source"] = "unreadable"
            else:
                y, sr = sf.read(p, dtype="float32")
                y = y if y.ndim == 1 else y.mean(1)
                restored = sidon(y, sr, 0.0, len(y) / sr)
                row["end_drop_db"] = round(ending_drop_db(restored, SR_SAVE, y, sr), 2)
                if row["end_drop_db"] < -a.max_end_drop:
                    row["source"] = "fallback"
                else:
                    wav = out / p.name
                    sf.write(wav, loudness_normalize(restored, SR_SAVE, a.target_lufs, a.peak), SR_SAVE, subtype="PCM_16")
                    ref = for_cer(text)
                    row["cer_sidon"] = round(cer(wav, ref), 4)
                    row["source"], co = word_check(row["cer_sidon"], None, lambda: cer(p, ref), a.first_gate, a.max_extra)
                    row["cer_orig"] = None if co is None else round(co, 4)
                    if row["source"] == "sidon":
                        (out / txt.name).write_text(txt.read_text(encoding="utf-8"), encoding="utf-8")
                    else:
                        wav.unlink()
            n[row["source"]] += 1
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 200 == 0 or i == len(todo):
                fh.flush()
                print(f"[{i}/{len(todo)}] {n}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
