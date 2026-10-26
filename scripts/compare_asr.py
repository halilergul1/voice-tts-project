"""Compare two ASR result files for the same episode (e.g. per-clip vs packed inference).

usage: .venv/bin/python scripts/compare_asr.py data/segments/trt/<uid>/asr.jsonl data/segments/trt/<uid>/asr_packed.jsonl [--max-cer 0.2]
"""
import argparse
import json
import sys

import numpy as np


def load(p):
    rows = {}
    for line in open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--max-cer", type=float, default=0.2)
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    common = sorted(set(A) & set(B))
    print(f"ids: a={len(A)} b={len(B)} common={len(common)}")
    keep = lambda r: r.get("trim_start") is not None and r["cer"] <= args.max_cer and not r.get("error")
    ka = np.array([keep(A[i]) for i in common]); kb = np.array([keep(B[i]) for i in common])
    print(f"kept: a={ka.sum()} b={kb.sum()} | same decision {np.mean(ka == kb):.3%} | a-only {int((ka & ~kb).sum())} | b-only {int((~ka & kb).sum())}")
    both = [i for i in common if keep(A[i]) and keep(B[i])]
    if both:
        dc = np.array([B[i]["cer"] - A[i]["cer"] for i in both])
        ds = np.array([B[i]["trim_start"] - A[i]["trim_start"] for i in both])
        de = np.array([B[i]["trim_end"] - A[i]["trim_end"] for i in both])
        print(f"on {len(both)} clips kept by both: CER diff mean {dc.mean():+.4f} | |trim_start| diff median {np.median(np.abs(ds)):.3f}s p95 {np.percentile(np.abs(ds), 95):.3f}s | |trim_end| diff median {np.median(np.abs(de)):.3f}s p95 {np.percentile(np.abs(de), 95):.3f}s")
        print(f"edges differing by > 0.25 s: start {int((np.abs(ds) > 0.25).sum())} end {int((np.abs(de) > 0.25).sum())}")
    print("errors: a", sum(bool(r.get("error")) for r in A.values()), "b", sum(bool(r.get("error")) for r in B.values()))
    for i in [i for i in common if keep(A[i]) != keep(B[i])][:6]:
        print(f"  {i}: a cer {A[i]['cer']} '{A[i].get('span_text')}' | b cer {B[i]['cer']} '{B[i].get('span_text')}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
