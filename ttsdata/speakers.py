"""Speaker ids for the built dataset: embed every clip, cluster per episode, merge clusters across a series.

The corpus has no speaker labels; subtitles only name off-screen speakers (`speaker_hint`, ~6% of clips).
This stage derives a `speaker_id` per clip from speaker embeddings:

1. embed: one embedding per clip (ECAPA-TDNN, `speechbrain/spkrec-ecapa-voxceleb`, 16 kHz, unit norm),
   cached per episode as data/speakers/<uid>.npy + <uid>.json (ids, durations, model) so re-runs skip
   episodes already embedded and only new / changed clips are computed.
2. cluster: within each episode, agglomerative clustering (average linkage, cosine distance) cut at
   `speaker_threshold`.
3. merge: within a series, episode clusters are merged greedily by the cosine distance between their
   size-weighted centroids (`speaker_merge_threshold`), closest pair first. Two clusters of the same
   episode never merge: the per-episode step already decided they are apart, and splitting one voice in
   two ids is cheaper for TTS than fusing two voices under one id.
4. ids: `<series_slug>_spkNN`, numbered by cluster size within the series (spk00 = most clips).

Sanity check against the weak labels: for clips with a named `speaker_hint`, purity (clips in a cluster
that carry its majority hint) and completeness (clips of a hint that sit in its majority cluster),
optionally swept over thresholds (`ttsdata speakers --sweep 0.4,0.5,0.6`).
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.signal import resample_poly
from tqdm import tqdm

from .config import Config

EMBED_SR = 16000


def speakers_dir(cfg: Config) -> Path:
    return cfg.root / "speakers"


# ----------------------------------------------------------------------------------------------
# clustering logic (pure numpy / scipy; unit-tested with synthetic embeddings)
# ----------------------------------------------------------------------------------------------

def l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def _relabel(lab: np.ndarray) -> np.ndarray:
    """Relabel to 0..k-1 in order of first appearance."""
    seen: dict[int, int] = {}
    return np.array([seen.setdefault(int(v), len(seen)) for v in lab], dtype=int)


def cluster_embeddings(emb: np.ndarray, threshold: float) -> np.ndarray:
    """Average-linkage agglomerative clustering on cosine distance, cut at `threshold`.

    Returns an int label per row (0..k-1, numbered by first appearance)."""
    n = len(emb)
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    z = linkage(l2norm(emb), method="average", metric="cosine")
    return _relabel(fcluster(z, t=threshold, criterion="distance"))


def centroids(emb: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit-norm centroid and size of every cluster, indexed by label."""
    k = int(labels.max()) + 1 if len(labels) else 0
    c = np.zeros((k, emb.shape[1]), dtype=np.float64)
    np.add.at(c, labels, l2norm(emb))
    return l2norm(c), np.bincount(labels, minlength=k)


def merge_clusters(cents: np.ndarray, sizes: np.ndarray, groups: list, threshold: float) -> np.ndarray:
    """Greedy agglomerative merge of per-episode clusters into series-level clusters.

    cents: (k, d) centroids, sizes: (k,) clip counts, groups: (k,) episode of each cluster.
    Repeatedly merges the closest pair (cosine distance between size-weighted centroids) whose
    episode sets are disjoint, until the closest allowed pair is farther than `threshold`.
    Returns a series-level label per input cluster (0..m-1, numbered by first appearance)."""
    k = len(cents)
    if k == 0:
        return np.zeros(0, dtype=int)
    gid = {g: i for i, g in enumerate(dict.fromkeys(groups))}
    ep = np.zeros((k, len(gid)), dtype=bool)
    ep[np.arange(k), [gid[g] for g in groups]] = True
    vec = l2norm(cents) * np.asarray(sizes, dtype=np.float64)[:, None]
    u = l2norm(vec)
    d = 1.0 - u @ u.T
    d[(ep.astype(int) @ ep.T.astype(int)) > 0] = np.inf  # same episode (incl. the diagonal): never merge
    members: dict[int, list[int]] = {i: [i] for i in range(k)}
    alive = np.ones(k, dtype=bool)
    while True:
        i, j = divmod(int(np.argmin(d)), k)
        if not np.isfinite(d[i, j]) or d[i, j] > threshold:
            break
        vec[i] += vec[j]
        ep[i] |= ep[j]
        members[i] += members.pop(j)
        alive[j] = False
        d[j, :] = np.inf
        d[:, j] = np.inf
        u[i] = l2norm(vec[i])
        row = 1.0 - u @ u[i]
        row[~alive] = np.inf
        row[i] = np.inf
        row[(ep & ep[i]).any(axis=1)] = np.inf  # merged group now shares an episode with these
        d[i, :] = row
        d[:, i] = row
    labels = np.empty(k, dtype=int)
    for new, rep in enumerate(sorted(members, key=lambda r: min(members[r]))):
        labels[members[rep]] = new
    return labels


def speaker_ids(slug: str, labels: np.ndarray) -> list[str]:
    """`<slug>_spkNN` per row, numbered by cluster size (largest first, ties by label)."""
    sizes = Counter(int(v) for v in labels)
    order = sorted(sizes, key=lambda lab: (-sizes[lab], lab))
    width = max(2, len(str(max(len(order) - 1, 0))))
    name = {lab: f"{slug}_spk{r:0{width}d}" for r, lab in enumerate(order)}
    return [name[int(v)] for v in labels]


_TR = str.maketrans("çğıöşüâîûÇĞİÖŞÜÂÎÛ", "cgiosuaiuCGIOSUAIU")


def series_slug(series: str) -> str:
    """'Eşref Rüya' -> 'esrefruya' (matches the Kanal D uid prefix), 'Payitaht Abdülhamid' -> 'payitahtabdulhamid'."""
    s = unicodedata.normalize("NFKD", series.translate(_TR)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower()) or "series"


# subtitle labels that name a role / gender rather than one character
GENERIC_HINTS = {"erkek", "kadın", "adam", "çocuk", "kız", "polis", "***", "hep bir ağızdan", "herkes", "hepsi",
                 "tv", "televizyon", "radyo", "spiker", "sunucu", "anons", "ses", "dış ses", "iç ses"}
_HINT_SUFFIX = re.compile(r"\s+(dış ses|iç ses|ses)$", re.IGNORECASE)


def normalize_hint(hint: str | None) -> str | None:
    """'Nisan ses' / 'Nisan dış ses' -> 'nisan'; 'Bidâr' -> 'bidar'; generic labels -> None."""
    if not hint:
        return None
    h = _HINT_SUFFIX.sub("", hint.strip()).strip().translate(str.maketrans("âîûÂÎÛ", "aiuAIU")).lower()
    if not h or h in GENERIC_HINTS or not re.search(r"[a-zçğıöşü]", h):
        return None
    return h


def purity_completeness(labels, hints) -> dict:
    """Over rows whose hint is not None: purity = sum over clusters of the majority-hint count / n,
    completeness = sum over hints of the majority-cluster count / n."""
    pairs = [(lab, h) for lab, h in zip(labels, hints) if h is not None]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "purity": float("nan"), "completeness": float("nan"), "hints": 0, "clusters": 0}
    by_cluster: dict = defaultdict(Counter)
    by_hint: dict = defaultdict(Counter)
    for lab, h in pairs:
        by_cluster[lab][h] += 1
        by_hint[h][lab] += 1
    return {"n": n,
            "purity": sum(max(c.values()) for c in by_cluster.values()) / n,
            "completeness": sum(max(c.values()) for c in by_hint.values()) / n,
            "hints": len(by_hint), "clusters": len(by_cluster)}


def assign_speakers(rows: list[dict], vectors: dict[str, np.ndarray], threshold: float,
                    merge_threshold: float) -> dict[str, tuple[str, int]]:
    """Cluster per episode, merge across each series, return {clip id: (speaker_id, cluster size)}."""
    by_series: dict = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        by_series[r["series"]][r["uid"]].append(i)
    out: dict[str, tuple[str, int]] = {}
    for series, episodes in by_series.items():
        cents, sizes, groups, owners = [], [], [], []
        for uid, idxs in episodes.items():
            emb = np.stack([vectors[rows[i]["id"]] for i in idxs])
            lab = cluster_embeddings(emb, threshold)
            c, s = centroids(emb, lab)
            for ci in range(len(s)):
                cents.append(c[ci])
                sizes.append(int(s[ci]))
                groups.append(uid)
                owners.append([idxs[j] for j in np.flatnonzero(lab == ci)])
        merged = merge_clusters(np.stack(cents), np.array(sizes), groups, merge_threshold)
        row_idx, row_lab = [], []
        for ci, m in enumerate(merged):
            row_idx += owners[ci]
            row_lab += [int(m)] * len(owners[ci])
        ids = speaker_ids(series_slug(series), np.array(row_lab))
        size = Counter(ids)
        for i, sid in zip(row_idx, ids):
            out[rows[i]["id"]] = (sid, size[sid])
    return out


# ----------------------------------------------------------------------------------------------
# embeddings
# ----------------------------------------------------------------------------------------------

def load_encoder(cfg: Config):
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    device = cfg.speaker_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    savedir = cfg.root / "models" / cfg.speaker_model.replace("/", "--")
    enc = EncoderClassifier.from_hparams(source=cfg.speaker_model, savedir=str(savedir), run_opts={"device": device})
    enc.eval()
    return enc, device


def _load_16k(path: Path) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != EMBED_SR:
        g = gcd(sr, EMBED_SR)
        y = resample_poly(y, EMBED_SR // g, sr // g).astype(np.float32)
    return y


def embed_clips(paths: list[Path], durations: list[float], enc, device: str, batch: int,
                desc: str = "embedding") -> np.ndarray:
    """One unit-norm embedding per path; clips are batched by length so padding stays small."""
    import torch

    n = len(paths)
    order = sorted(range(n), key=lambda i: durations[i])
    out: list = [None] * n
    with tqdm(total=n, desc=desc, unit="clip", leave=False) as bar:
        for b in range(0, n, batch):
            idx = order[b:b + batch]
            wavs = [torch.from_numpy(_load_16k(paths[i])) for i in idx]
            lens = torch.tensor([max(len(w), 1) for w in wavs], dtype=torch.float32)
            t = int(lens.max())
            x = torch.zeros(len(wavs), t)
            for j, w in enumerate(wavs):
                x[j, :len(w)] = w
            with torch.no_grad():
                e = enc.encode_batch(x.to(device), (lens / t).to(device)).squeeze(1).float().cpu().numpy()
            for j, i in enumerate(idx):
                out[i] = e[j]
            bar.update(len(idx))
    return l2norm(np.stack(out)).astype(np.float32)


def embed_episode(uid: str, rows: list[dict], cfg: Config, encoder) -> dict[str, np.ndarray]:
    """Embeddings for every clip of one episode, from cache where ids / durations / model match.

    `encoder` is a zero-arg callable returning (model, device); it is only called when something is missing."""
    sdir = speakers_dir(cfg)
    sdir.mkdir(parents=True, exist_ok=True)
    npy, meta_file = sdir / f"{uid}.npy", sdir / f"{uid}.json"
    cached: dict[str, tuple[float, np.ndarray]] = {}
    if npy.exists() and meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        emb = np.load(npy)
        if meta.get("model") == cfg.speaker_model and len(meta.get("ids", [])) == len(emb):
            cached = {i: (d, emb[k]) for k, (i, d) in enumerate(zip(meta["ids"], meta["durations"]))}
    todo = [r for r in rows if r["id"] not in cached or cached[r["id"]][0] != r["duration"]]
    if todo:
        enc, device = encoder()
        new = embed_clips([cfg.dataset_dir / r["audio_file"] for r in todo], [r["duration"] for r in todo],
                          enc, device, cfg.speaker_batch, desc=uid)
        for r, e in zip(todo, new):
            cached[r["id"]] = (r["duration"], e)
    ids = [r["id"] for r in rows]
    vectors = {i: cached[i][1] for i in ids}
    if todo or list(cached) != ids:
        np.save(npy, np.stack([vectors[i] for i in ids]).astype(np.float32))
        meta_file.write_text(json.dumps({"uid": uid, "model": cfg.speaker_model, "ids": ids,
                                         "durations": [r["duration"] for r in rows]}), encoding="utf-8")
    return vectors


def embed_all(cfg: Config, rows: list[dict]) -> dict[str, np.ndarray]:
    by_uid: dict = defaultdict(list)
    for r in rows:
        by_uid[r["uid"]].append(r)
    state: dict = {}

    def encoder():
        if "enc" not in state:
            state["enc"] = load_encoder(cfg)
            print(f"[speakers] {cfg.speaker_model} on {state['enc'][1]}")
        return state["enc"]

    vectors: dict[str, np.ndarray] = {}
    for uid, ep_rows in by_uid.items():
        before = len(vectors)
        vectors.update(embed_episode(uid, ep_rows, cfg, encoder))
        print(f"[ok] {uid}: {len(vectors) - before} clips embedded" if "enc" in state else f"[skip] {uid} cached")
    return vectors


# ----------------------------------------------------------------------------------------------
# evaluation, writing, driver
# ----------------------------------------------------------------------------------------------

def load_rows(cfg: Config) -> list[dict]:
    p = cfg.dataset_dir / "metadata.jsonl"
    if not p.exists():
        raise SystemExit(f"{p} not found: run `ttsdata build` first")
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(rows: list[dict], assign: dict[str, tuple[str, int]]) -> dict:
    """Overall and per-series hint purity / completeness plus cluster-count summary."""
    hints = [normalize_hint(r.get("speaker_hint")) for r in rows]
    labels = [assign[r["id"]][0] for r in rows]
    overall = purity_completeness(labels, hints)
    overall.update(_size_summary(labels))
    per_series = {}
    for series in dict.fromkeys(r["series"] for r in rows):
        idx = [i for i, r in enumerate(rows) if r["series"] == series]
        m = purity_completeness([labels[i] for i in idx], [hints[i] for i in idx])
        m.update(_size_summary([labels[i] for i in idx]))
        per_series[series] = m
    return {"overall": overall, "per_series": per_series}


def _size_summary(labels: list) -> dict:
    """Cluster-count summary: how many ids, how many usable ones, and what share of clips they hold."""
    sizes = Counter(labels)
    n = len(labels)
    big = [v for v in sizes.values() if v >= 10]
    return {"clips": n, "speakers": len(sizes), "speakers_ge10": len(big),
            "coverage_ge10": round(sum(big) / n, 3) if n else 0.0,  # share of clips whose id has >= 10 clips
            "largest": max(sizes.values()) if sizes else 0,
            "largest_share": round(max(sizes.values()) / n, 3) if n else 0.0,
            "median_size": float(np.median(list(sizes.values()))) if sizes else 0.0}


def _fmt(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def sweep_table(results: list[tuple[float, float, dict]]) -> str:
    lines = ["| threshold | merge | purity | completeness | hinted clips | hint names | hinted clusters | speakers | speakers >=10 clips | clips covered by them | median size |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t, m, ev in results:
        o = ev["overall"]
        lines.append(f"| {t:.2f} | {m:.2f} | {_fmt(o['purity'])} | {_fmt(o['completeness'])} | {o['n']} | {o['hints']} | "
                     f"{o['clusters']} | {o['speakers']} | {o['speakers_ge10']} | {o['coverage_ge10']:.2f} | {o['median_size']:.0f} |")
    return "\n".join(lines)


def series_table(ev: dict) -> str:
    lines = ["| series | clips | speakers | >=10 clips | clips covered | largest share | hinted clips | hint names | purity | completeness |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for series, m in ev["per_series"].items():
        lines.append(f"| {series} | {m['clips']} | {m['speakers']} | {m['speakers_ge10']} | {m['coverage_ge10']:.2f} | "
                     f"{m['largest_share']:.2f} | {m['n']} | {m['hints']} | {_fmt(m['purity'])} | {_fmt(m['completeness'])} |")
    return "\n".join(lines)


def write_back(cfg: Config, rows: list[dict], assign: dict[str, tuple[str, int]], ev: dict) -> dict:
    """speaker_id / speaker_cluster_size into metadata.jsonl, a 4th column in the csv files, stats + README."""
    from .build import readme

    out = cfg.dataset_dir
    for r in rows:
        r["speaker_id"], r["speaker_cluster_size"] = assign[r["id"]]

    def lj_line(r):
        return f"{r['id']}|{r['text']}|{r['normalized_text']}|{r['speaker_id']}\n"

    with (out / "metadata.csv").open("w", encoding="utf-8") as f_all, \
         (out / "train.csv").open("w", encoding="utf-8") as f_tr, \
         (out / "val.csv").open("w", encoding="utf-8") as f_va, \
         (out / "metadata.jsonl").open("w", encoding="utf-8") as f_js:
        for r in rows:
            f_all.write(lj_line(r))
            (f_va if r.get("split") == "val" else f_tr).write(lj_line(r))
            f_js.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats_file = out / "stats.json"
    stats = json.loads(stats_file.read_text(encoding="utf-8")) if stats_file.exists() else {}
    stats["speakers"] = {
        "model": cfg.speaker_model, "threshold": cfg.speaker_threshold, "merge_threshold": cfg.speaker_merge_threshold,
        "total": ev["overall"]["speakers"], "hint_purity": round(ev["overall"]["purity"], 4),
        "hint_completeness": round(ev["overall"]["completeness"], 4), "hinted_clips": ev["overall"]["n"],
        "per_series": {s: {k: _json_num(v) for k, v in m.items()} for s, m in ev["per_series"].items()},
    }
    stats_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    if "utterances" in stats:
        (out / "README.md").write_text(readme(stats, cfg), encoding="utf-8")
    return stats["speakers"]


def _json_num(v):
    """Round floats for stats.json; NaN (no hinted clips in a series) becomes null."""
    if isinstance(v, float):
        return None if np.isnan(v) else round(v, 4)
    return v


def _parse_list(spec: str | None) -> list[float] | None:
    return [float(x) for x in spec.split(",") if x.strip()] if spec else None


def run(cfg: Config, sweep: str | None = None, sweep_merge: str | None = None, write: bool = True) -> dict:
    rows = load_rows(cfg)
    vectors = embed_all(cfg, rows)
    thresholds = _parse_list(sweep)
    if thresholds:
        merges = _parse_list(sweep_merge) or [cfg.speaker_merge_threshold] * len(thresholds)
        if len(merges) != len(thresholds):
            raise SystemExit("--sweep-merge must list as many values as --sweep")
        results = []
        for t, m in zip(thresholds, merges):
            ev = evaluate(rows, assign_speakers(rows, vectors, t, m))
            results.append((t, m, ev))
            print(f"[sweep] threshold={t:.2f} merge={m:.2f} purity={_fmt(ev['overall']['purity'])} "
                  f"completeness={_fmt(ev['overall']['completeness'])} speakers={ev['overall']['speakers']}")
        print()
        print(sweep_table(results))
        return {"sweep": [(t, m, ev["overall"]) for t, m, ev in results]}
    assign = assign_speakers(rows, vectors, cfg.speaker_threshold, cfg.speaker_merge_threshold)
    ev = evaluate(rows, assign)
    print(sweep_table([(cfg.speaker_threshold, cfg.speaker_merge_threshold, ev)]))
    print()
    print(series_table(ev))
    if write:
        st = write_back(cfg, rows, assign, ev)
        print(f"\nwrote speaker_id for {len(rows)} clips ({st['total']} speakers) to {cfg.dataset_dir}")
    return ev
