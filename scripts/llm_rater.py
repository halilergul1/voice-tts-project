#!/usr/bin/env python
"""Blind emotion rating of the calibration clips by an audio-input LLM (Gemini), graded against the human ratings.

    .venv/bin/python scripts/llm_rater.py --model gemini-2.5-flash --limit 3 --dry-run     # bodies only, no key
    .venv/bin/python scripts/llm_rater.py --list-models                                     # what the key can call
    .venv/bin/python scripts/llm_rater.py --model gemini-2.5-flash                          # all 300, resumable

Key: GEMINI_API_KEY in the environment, or --key-file PATH (a file holding only the key). It is sent in a header,
never printed, never written. Replies are cached per clip in --out, so a quota stop resumes where it left off.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttsdata.emotion import parse_ratings  # noqa: E402
from ttsdata.llmrater import (Pacer, backoff_seconds, parse_batch_response, parse_response, quota_kind, report,  # noqa: E402
                              request_body, request_body_batch)

API = "https://generativelanguage.googleapis.com/v1beta"


def load_key(a) -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key and a.key_file:
        key = Path(a.key_file).expanduser().read_text(encoding="utf-8").strip()
    if not key:
        raise SystemExit("no key: export GEMINI_API_KEY or pass --key-file")
    return key


def post(session, url: str, body: dict, key: str, pacer: Pacer, max_wait: float = 1800.0) -> tuple[dict | None, str]:
    """One paced request with retries: 429 slows the pacer down (a per-day quota stops the run), 5xx and transport
    errors back off; gives up after max_wait seconds of trying. Returns (response, reason-when-None)."""
    t0 = time.monotonic()
    attempt = 0
    while True:
        pacer.wait()
        try:
            r = session.post(url, headers={"x-goog-api-key": key, "Content-Type": "application/json"}, json=body, timeout=120)
        except requests.exceptions.RequestException as e:  # dropped connection, read timeout: same as a 503
            r = None
            note = type(e).__name__
        if r is not None and r.status_code == 200:
            pacer.recover()
            return r.json(), ""
        if r is not None and r.status_code == 429:
            try:
                kind = quota_kind(r.json())
            except ValueError:
                kind = "unknown"
            if kind == "day":
                return None, "daily quota reached"
            note = f"HTTP 429 ({kind} quota), gap now {pacer.slow_down():.0f} s"
        elif r is not None and r.status_code in (500, 503):
            note = f"HTTP {r.status_code}"
        elif r is not None:
            raise SystemExit(f"HTTP {r.status_code}: {r.text[:400]}")
        if time.monotonic() - t0 > max_wait:
            return None, f"gave up after {max_wait:.0f} s ({note})"
        ra = r.headers.get("Retry-After") if r is not None else None
        wait = backoff_seconds(attempt, float(ra) if ra else None)
        print(f"  {note}, retry in {wait:.0f} s", flush=True)
        time.sleep(wait)
        attempt += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--clips", default="data/dataset/calibrate_emotions")
    ap.add_argument("--ratings", default="calibration/emotion_ratings_2026-10-27.csv")
    ap.add_argument("--out", default="", help="cache jsonl; default <clips>/llm_<model>.jsonl")
    ap.add_argument("--emotions-roots", default="data/dataset,data/dataset_long", help="emotions.jsonl roots for the comparison")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rpm", type=float, default=8.0, help="requests per minute ceiling (free tier is low)")
    ap.add_argument("--workers", type=int, default=1, help="parallel requests in flight (each call can take 30 s)")
    ap.add_argument("--per-request", type=int, default=1, help="clips per request (free tier: 20 requests a day per model)")
    ap.add_argument("--thinking", default="", help="Gemini 3.x thinking: low / minimal / off (thinkingBudget 0); '' leaves the model default")
    ap.add_argument("--key-file", default="")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build the requests, call nothing")
    a = ap.parse_args()

    session = requests.Session()
    if a.list_models:
        r = session.get(f"{API}/models", headers={"x-goog-api-key": load_key(a)}, params={"pageSize": 200}, timeout=60)
        r.raise_for_status()
        for mdl in r.json().get("models", []):
            if "generateContent" in mdl.get("supportedGenerationMethods", []):
                print(mdl["name"].removeprefix("models/"), "-", mdl.get("displayName", ""))
        return

    clips = Path(a.clips)
    sheet = list(csv.DictReader(clips.joinpath("ratings.csv").open(encoding="utf-8")))  # id -> file, as exported
    files = {r["id"].strip(): r["file"] for r in sheet}
    ratings = parse_ratings(Path(a.ratings).read_text(encoding="utf-8"))
    todo = [cid for cid in files if cid in ratings]
    out = Path(a.out) if a.out else clips / f"llm_{a.model}.jsonl"
    done = {}
    if out.exists():
        done = {json.loads(l)["id"]: json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    pending = [cid for cid in todo if cid not in done]
    if a.limit:
        pending = pending[: max(0, a.limit - len(done))]
    print(f"{len(todo)} rated clips, {len(done)} cached, {len(pending)} to send to {a.model}")

    if pending and not a.dry_run:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        key = load_key(a)
        url = f"{API}/models/{a.model}:generateContent"
        pacer = Pacer(a.rpm)
        stop = {"reason": ""}

        def rate(group):
            """One request for a group of clip ids -> (group, parsed rows or None)."""
            if stop["reason"]:
                return group, None
            if len(group) == 1:
                body = request_body((clips / files[group[0]]).read_bytes(), "audio/wav")
            else:
                body = request_body_batch([(cid, (clips / files[cid]).read_bytes()) for cid in group], "audio/wav")
            if a.thinking == "off":
                body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}   # 3 s per clip instead of 30
            elif a.thinking:
                body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": a.thinking}
            resp, reason = post(session, url, body, key, pacer)
            if resp is None:
                stop["reason"] = reason
                return group, None
            return group, [parse_response(resp)] if len(group) == 1 else parse_batch_response(resp, len(group))

        groups = [pending[i:i + a.per_request] for i in range(0, len(pending), a.per_request)]
        n = 0
        with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=a.workers) as pool:
            for fut in as_completed([pool.submit(rate, g) for g in groups]):
                group, parsed = fut.result()
                if parsed is None:
                    continue
                for cid, p in zip(group, parsed):
                    if p["label"] is None and len(group) > 1:
                        print(f"  {cid}: no answer in the batch reply, left for a re-run", flush=True)
                        continue   # not cached: the next run asks again
                    row = {"id": cid, "file": files[cid], "model": a.model, **p}
                    done[cid] = row
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n += 1
                    print(f"  {n}/{len(pending)} {cid}: {row['label']} {row['conf']} '{row['style']}'", flush=True)
                f.flush()
        if stop["reason"]:
            print(f"stopped: {stop['reason']}; re-run later, the cache resumes")
    elif pending:
        sizes = [len(json.dumps(request_body((clips / files[c]).read_bytes(), "audio/wav"))) for c in pending[:5]]
        print(f"dry run: first bodies {[round(s / 1024) for s in sizes]} KB; nothing sent")
        return

    rows = [done[c] for c in todo if c in done]
    if not rows:
        return
    e2v = {}
    for root in a.emotions_roots.split(","):
        p = Path(root) / "emotions.jsonl"
        if p.exists():
            for l in p.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    d = json.loads(l)
                    e2v[d["id"]] = (d.get("e2v_label"), float(d.get("conf") or 0.0))
    text = report(a.model, rows, e2v, ratings, cost_tokens=sum(r.get("tokens") or 0 for r in rows))
    md = clips / f"LLM_RATER_{a.model}.md"
    md.write_text(text, encoding="utf-8")
    print(text)
    print(f"report: {md}")


if __name__ == "__main__":
    main()
