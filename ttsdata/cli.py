"""Command line entry point.

  ttsdata download-trt   --ids ID ...  | --csv meta.csv --series REGEX --limit N
  ttsdata download-kanald --slug esrefruya --name "Eşref Rüya" --url-template "https://cdn.duhnet.tv/C13/2013kanald/dizi/esrefruya/er{n}.mp4" --episodes 1-10
  ttsdata segment        [--only UID ...]
  ttsdata verify         [--only UID ...] [--model REPO]
  ttsdata build          [--max-cer 0.2 ...]
  ttsdata speakers       [--sweep 0.4,0.5,0.6] [--dry-run]
  ttsdata merge-clips    [--limit-episodes N]
  ttsdata soundtags      [--listen 4]
  ttsdata extend-events  [--limit-episodes N]
  ttsdata verify-events  [--limit N]
  ttsdata emotions       [--limit N] [--calibration 300] [--ratings ratings.csv] [--eiv]
  ttsdata stats
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import fields
from pathlib import Path

from .config import Config


def _parse_range(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _cfg_from_args(args) -> Config:
    cfg = Config(root=Path(args.root))
    for f in fields(Config):
        v = getattr(args, f.name, None)
        if v is not None and f.name != "root":
            setattr(cfg, f.name, v)
    return cfg


def _add_cfg_overrides(p: argparse.ArgumentParser) -> None:
    for f in fields(Config):
        if f.name == "root":
            continue
        typ = {bool: lambda s: s.lower() in ("1", "true", "yes")}.get(f.type if isinstance(f.type, type) else str, None)
        if f.type in ("int", int):
            typ = int
        elif f.type in ("float", float):
            typ = float
        elif f.type in ("str", str):
            typ = str
        elif f.type in ("bool", bool):
            typ = lambda s: s.lower() in ("1", "true", "yes")
        p.add_argument(f"--{f.name.replace('_', '-')}", dest=f.name, type=typ, default=None, help=argparse.SUPPRESS)


def cmd_download_trt(args) -> None:
    from .sources import download_youtube

    cfg = _cfg_from_args(args)
    ids: list[str] = list(args.ids or [])
    if args.csv:
        import pandas as pd

        df = pd.read_csv(args.csv)
        if args.series:
            df = df[df["title"].str.contains(args.series, case=False, regex=True, na=False)]
        if args.min_minutes:
            df = df[df["duration_seconds"] >= args.min_minutes * 60]
        cand = df["id"].tolist()
        if args.shuffle:
            random.Random(cfg.seed).shuffle(cand)
        if args.shard:
            k, n = (int(x) for x in args.shard.split("/"))
            cand = cand[k::n]  # shard k of n, for several downloaders in parallel
        if args.limit:
            cand = cand[: args.limit]
        ids.extend(cand)
    done = 0
    for vid in ids:
        try:
            if download_youtube(vid, cfg.raw_dir):
                done += 1
                print(f"[ok] {vid}")
        except Exception as e:
            print(f"[fail] {vid}: {e}")
    print(f"downloaded {done}/{len(ids)}")


def cmd_download_kanald(args) -> None:
    from .sources import download_kanald

    cfg = _cfg_from_args(args)
    done = 0
    eps = _parse_range(args.episodes)
    for n in eps:
        url = args.url_template.format(n=n)
        try:
            if download_kanald(args.slug, args.name, n, url, cfg.raw_dir):
                done += 1
                print(f"[ok] {args.slug} ep{n}")
        except Exception as e:
            print(f"[fail] {args.slug} ep{n}: {e}")
    print(f"downloaded {done}/{len(eps)}")


def cmd_segment(args) -> None:
    import time

    from .locks import try_lock, unlock
    from .segment import cut_episode
    from .sources import iter_raw

    cfg = _cfg_from_args(args)
    for raw in iter_raw(cfg.raw_dir):
        info = json.loads((raw / "info.json").read_text(encoding="utf-8"))
        if args.only and info["uid"] not in args.only:
            continue
        out = cfg.seg_dir / info["source"] / info["uid"]
        if (out / "segments.jsonl").exists() and not args.force:
            print(f"[skip] {info['uid']} already segmented")
            continue
        if not try_lock(out, ".segment.lock"):
            continue  # another segmenter is on it
        try:
            t0 = time.time()
            st = cut_episode(raw, cfg)
            st["seconds"] = round(time.time() - t0, 1)
            print(json.dumps(st, ensure_ascii=False), flush=True)
        finally:
            unlock(out, ".segment.lock")


def cmd_verify(args) -> None:
    from .asr import verify_episode

    cfg = _cfg_from_args(args)
    for seg_file in sorted(cfg.seg_dir.glob("*/*/segments.jsonl")):
        if args.only and seg_file.parent.name not in args.only:
            continue
        st = verify_episode(seg_file.parent, cfg, candidates_only=not args.all_segments, asr_filename=args.asr_file)
        print(json.dumps(st, ensure_ascii=False))


def cmd_build(args) -> None:
    from .build import build

    cfg = _cfg_from_args(args)
    st = build(cfg)
    print(json.dumps({k: v for k, v in st.items() if k not in ("per_series",)}, ensure_ascii=False, indent=2))
    print(f"dataset at {cfg.dataset_dir}")


def cmd_speakers(args) -> None:
    from .speakers import run

    cfg = _cfg_from_args(args)
    run(cfg, sweep=args.sweep, sweep_merge=args.sweep_merge, write=not (args.dry_run or args.sweep))


def cmd_merge_clips(args) -> None:
    from .clipmerge import run

    cfg = _cfg_from_args(args)
    st = run(cfg, limit_episodes=args.limit_episodes)
    print(json.dumps({k: v for k, v in st.items() if k != "per_series"}, ensure_ascii=False, indent=2))


def cmd_soundtags(args) -> None:
    from .soundtags import run

    cfg = _cfg_from_args(args)
    run(cfg, listen=args.listen)


def cmd_extend_events(args) -> None:
    from .eventclips import run

    cfg = _cfg_from_args(args)
    st = run(cfg, limit_episodes=args.limit_episodes)
    print(json.dumps(st, ensure_ascii=False, indent=2))


def cmd_verify_events(args) -> None:
    from .eventverify import run

    cfg = _cfg_from_args(args)
    st = run(cfg, limit=args.limit)
    print(json.dumps(st, ensure_ascii=False, indent=2))


def cmd_emotions(args) -> None:
    from .emotion import run

    cfg = _cfg_from_args(args)
    st = run(cfg, limit=args.limit, calibration=args.calibration, ratings=args.ratings, eiv=args.eiv,
             eiv_all=args.eiv_all, require_agreement=args.require_agreement)
    print(json.dumps(st, ensure_ascii=False, indent=2, default=str))


def cmd_stats(args) -> None:
    cfg = _cfg_from_args(args)
    p = cfg.dataset_dir / "stats.json"
    if not p.exists():
        sys.exit("no dataset built yet")
    print(p.read_text(encoding="utf-8"))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="ttsdata", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data", help="data root (default: data)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download-trt", help="fetch YouTube audio + Turkish subtitles")
    s.add_argument("--ids", nargs="*", help="YouTube video ids")
    s.add_argument("--csv", help="metadata csv with id,title,duration_seconds columns")
    s.add_argument("--series", help="regex on title, e.g. 'Seksenler|Payitaht'")
    s.add_argument("--limit", type=int)
    s.add_argument("--min-minutes", type=float, default=None)
    s.add_argument("--shuffle", action="store_true")
    s.add_argument("--shard", help="k/n: take every n-th id starting at k (run n downloaders in parallel)")
    _add_cfg_overrides(s)
    s.set_defaults(func=cmd_download_trt)

    s = sub.add_parser("download-kanald", help="fetch Kanal D Engelsiz audio + srt from the CDN")
    s.add_argument("--slug", required=True, help="short id used in uids, e.g. esrefruya")
    s.add_argument("--name", required=True, help="series display name")
    s.add_argument("--url-template", required=True, help="mp4 url with {n} for the episode number")
    s.add_argument("--episodes", required=True, help="e.g. 1-10 or 1,3,5")
    _add_cfg_overrides(s)
    s.set_defaults(func=cmd_download_kanald)

    s = sub.add_parser("segment", help="cut subtitle-aligned, VAD-trimmed segments")
    s.add_argument("--only", nargs="*")
    s.add_argument("--force", action="store_true")
    _add_cfg_overrides(s)
    s.set_defaults(func=cmd_segment)

    s = sub.add_parser("verify", help="transcribe segments with Whisper and score against subtitles")
    s.add_argument("--only", nargs="*")
    s.add_argument("--all-segments", action="store_true", help="also transcribe clips outside the duration window")
    s.add_argument("--asr-file", default="asr.jsonl", help="output file name per episode (experiments)")
    _add_cfg_overrides(s)
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("build", help="filter + normalise + write the final dataset")
    _add_cfg_overrides(s)
    s.set_defaults(func=cmd_build)

    s = sub.add_parser("speakers", help="embed clips, cluster per episode, merge across series, add speaker_id")
    s.add_argument("--sweep", help="comma list of speaker_threshold values to score against speaker_hint (no write)")
    s.add_argument("--sweep-merge", help="comma list of speaker_merge_threshold values zipped with --sweep "
                                         "(default: the configured merge threshold for every row)")
    s.add_argument("--dry-run", action="store_true", help="cluster and report only; do not touch the dataset")
    _add_cfg_overrides(s)
    s.set_defaults(func=cmd_speakers)

    s = sub.add_parser("merge-clips", help="join adjacent same-voice lines into longer clips (data/dataset_long)")
    s.add_argument("--limit-episodes", type=int, default=0, help="only the first N episodes; writes data/dataset_long_smoke/ (smoke tests)")
    _add_cfg_overrides(s)
    s.set_defaults(func=cmd_merge_clips)

    s = sub.add_parser("soundtags", help="mine subtitle sound descriptions as style / non-verbal labels and flags")
    s.add_argument("--listen", type=int, default=4, help="clips per tag in data/dataset/listen_tags/")
    _add_cfg_overrides(s)
    s.set_defaults(func=cmd_soundtags)

    s = sub.add_parser("extend-events", help="re-cut clips to include the adjacent laugh / cry / sigh, tagged in the text (data/dataset_events)")
    s.add_argument("--limit-episodes", type=int, default=0, help="only the first N episodes; writes data/dataset_events_smoke/")
    _add_cfg_overrides(s)
    s.set_defaults(func=cmd_extend_events)

    s = sub.add_parser("verify-events", help="confirm the tagged sound in each extended clip and reject narrator speech (Whisper + PANNs)")
    s.add_argument("--limit", type=int, default=0, help="only the first N rows (smoke tests)")
    _add_cfg_overrides(s)
    s.set_defaults(func=cmd_verify_events)

    s = sub.add_parser("emotions", help="coarse emotion labels per clip (emotion2vec+), calibration export and scoring")
    s.add_argument("--limit", type=int, default=0, help="only the first N clips per root (smoke tests)")
    s.add_argument("--calibration", type=int, default=0, help="export N clips for native raters (data/dataset/calibrate_emotions)")
    s.add_argument("--ratings", default=None, help="score a filled ratings.csv instead of labelling")
    s.add_argument("--eiv", action="store_true", help="also score the calibration clips with EmpathicInsight-Voice (experimental)")
    s.add_argument("--eiv-all", action="store_true", help="score every clip with EmpathicInsight-Voice (slow on CPU)")
    s.add_argument("--require-agreement", action="store_true", help="label only where both scorers agree (needs --eiv-all)")
    _add_cfg_overrides(s)
    s.set_defaults(func=cmd_emotions)

    s = sub.add_parser("stats", help="print stats.json of the built dataset")
    _add_cfg_overrides(s)
    s.set_defaults(func=cmd_stats)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
