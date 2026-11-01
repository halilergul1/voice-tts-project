"""Narrator emotion labels (stage 2 Task 3B): parse the describer's spans, link each emotion to the lines of the
character it names, write narration_styles.jsonl and NARRATION.md.

  .venv/bin/python scripts/narration_labels.py [--root data/dataset] [--episodes-total 1017]

Inputs under --root: narration.jsonl and narration_stats.jsonl (scripts/transcribe_narration.py), metadata.jsonl
(id, uid, series, orig_start, speaker_id, speaker_hint), emotions.jsonl (emotion2vec+ posteriors, for agreement).
Per episode: descriptions with an emotion word (narration.parse_description); the name resolves to a speaker cluster
through the series' subtitle speaker hints (narration.name_clusters; full name, else its first word, case-folded). A
resolved description labels that cluster's lines within [end - 2 s, end + 20 s] and before the same character's next
description (source "narration"); otherwise only the first line within 5 s after it (source "narration_noname").
Writes <root>/narration_styles.jsonl {id, style, source, desc, dt} (dt = line start - description end) and
<root>/NARRATION.md (narrator time, Whisper speed, descriptions and linked lines per emotion, name resolution,
agreement with emotion2vec+, a projection to --episodes-total episodes).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttsdata.narration import link_lines, name_clusters, parse_description  # noqa: E402
from ttsdata.soundtags import tr_lower  # noqa: E402

E2V_CLASS = {"angry": "angry", "happy": "happy", "sad": "sad", "crying": "sad", "scared": "fearful",
             "surprised": "surprised"}


def _load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def resolver(rows: list[dict]):
    """(series, name) -> speaker_id, matching names case-folded: the full name first, else its first word."""
    by_key = {(s, tr_lower(n)): spk for (s, n), spk in name_clusters(rows).items()}

    def resolve(series: str, name: str | None) -> str | None:
        if not name:
            return None
        low = tr_lower(name)
        return by_key.get((series, low)) or by_key.get((series, low.split()[0]))
    return resolve


def label_episode(spans: list[dict], lines: list[dict], series: str, resolve) -> tuple[list[dict], list[dict]]:
    """(descriptions, label rows) of one episode."""
    descs = []
    for sp in sorted(spans, key=lambda s: s["start"]):
        for d in parse_description(sp["text"]):
            descs.append({**d, "start": sp["start"], "end": sp["end"], "text": sp["text"],
                          "cluster": resolve(series, d["name"])})
    for k, d in enumerate(descs):  # the same character's next description closes the window
        nxt = next((e["start"] for e in descs[k + 1:] if d["cluster"] and e["cluster"] == d["cluster"]), None)
        if nxt is not None:
            d["next"] = nxt
    start_of = {l["id"]: l["start"] for l in lines}
    out = []
    for d in descs:
        source = "narration" if d["cluster"] else "narration_noname"
        for i in link_lines(d, lines, d["cluster"]):
            out.append({"id": i, "style": d["emotion"], "source": source, "desc": d["text"],
                        "dt": round(start_of[i] - d["end"], 2)})
    return descs, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/dataset")
    ap.add_argument("--episodes-total", type=int, default=1017, help="TRT episodes, for the projection")
    a = ap.parse_args()
    root = Path(a.root)
    spans = _load(root / "narration.jsonl")
    stats = _load(root / "narration_stats.jsonl")
    meta = _load(root / "metadata.jsonl")
    e2v = {e["id"]: e.get("e2v") or {} for e in _load(root / "emotions.jsonl")}
    resolve = resolver(meta)
    lines_by_uid: dict[str, list[dict]] = defaultdict(list)
    series_of = {}
    for r in meta:
        if r.get("orig_start") is not None:
            lines_by_uid[r["uid"]].append({"id": r["id"], "start": r["orig_start"], "speaker_id": r.get("speaker_id")})
            series_of[r["uid"]] = r.get("series")
    spans_by_uid: dict[str, dict] = defaultdict(dict)
    for s in spans:  # a rerun after a crash may have written an episode twice
        spans_by_uid[s["uid"]][(s["start"], s["end"])] = s
    descs, rows = [], []
    for uid, sp in spans_by_uid.items():
        d, r = label_episode(list(sp.values()), lines_by_uid.get(uid, []), series_of.get(uid), resolve)
        descs += d
        rows += r
    with (root / "narration_styles.jsonl").open("w", encoding="utf-8") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    n_ep = len({s["uid"] for s in stats}) or len(spans_by_uid)
    gap_sec = sum(s["gap_sec"] for s in stats)
    narr_sec = sum(s["narration_sec"] for s in stats)
    wall = sum(s.get("wall_sec", 0) for s in stats)
    audio = sum(s["audio_sec"] for s in stats)
    per_emo_desc = Counter(d["emotion"] for d in descs)
    named = [d for d in descs if d["name"]]
    resolved = [d for d in named if d["cluster"]]
    linked: dict[str, set] = defaultdict(set)
    by_source: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        linked[r["style"]].add(r["id"])
        by_source[r["style"]][r["source"]] += 1
    styles_of: dict[str, set] = defaultdict(set)
    for r in rows:
        styles_of[r["id"]].add(r["style"])
    conflicts = sum(1 for s in styles_of.values() if len({"sad" if x == "crying" else x for x in s}) > 1)
    scale = a.episodes_total / n_ep if n_ep else 0
    L = ["# Narrator emotion labels (Task 3B)", "",
         f"{n_ep} episodes, {audio / 3600:.1f} h audio; gaps between speech cues {gap_sec / 3600:.1f} h; narrator spans "
         f"{narr_sec / 3600:.2f} h ({narr_sec / max(n_ep, 1):.0f} s per episode); Whisper {wall / 60:.1f} min wall, "
         f"real-time factor {wall / gap_sec:.3f} on the gaps" if gap_sec else f"{n_ep} episodes (no stats file)", "",
         f"Spans {len(spans)}, descriptions with an emotion word {len(descs)}; named {len(named)} "
         f"({len(named) / max(len(descs), 1):.0%}), name resolved to a speaker cluster {len(resolved)} "
         f"({len(resolved) / max(len(named), 1):.0%} of named). Lines with conflicting narrator emotions: {conflicts}.", "",
         f"| emotion | descriptions | linked lines | named-and-resolved rows | no-name rows | e2v top class agrees | "
         f"e2v ≥ 0.5 | projected lines ({a.episodes_total} episodes) |",
         "|---|---|---|---|---|---|---|---|"]
    for emo in ("angry", "happy", "sad", "crying", "scared", "surprised"):
        ids = linked.get(emo, set())
        cls = E2V_CLASS[emo]
        scored = [e2v[i] for i in ids if e2v.get(i)]
        top = sum(1 for p in scored if max(p, key=p.get) == cls)
        hi = sum(1 for p in scored if p.get(cls, 0) >= 0.5)
        L.append(f"| {emo} | {per_emo_desc.get(emo, 0)} | {len(ids)} | {by_source[emo]['narration']} | "
                 f"{by_source[emo]['narration_noname']} | {top / max(len(scored), 1):.0%} | {hi / max(len(scored), 1):.0%} | "
                 f"{len(ids) * scale:,.0f} |")
    L += ["", "Examples (description -> emotion, name, resolved cluster):", ""]
    for d in descs[:: max(1, len(descs) // 15)][:15]:
        L.append(f"- {d['text']} -> {d['emotion']}, {d['name']}, {d['cluster']}")
    (root / "NARRATION.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
