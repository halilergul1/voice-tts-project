"""Coarse emotion labels (angry / happy / sad / neutral) for the corpus, with a native-rater calibration loop.

Scorers:
- emotion2vec+ large (FunASR, `iic/emotion2vec_plus_large`): 9 classes angry, disgusted, fearful, happy, neutral,
  other, sad, surprised, unknown. Language-agnostic SSL features; used as the emotion judge by CosyVoice 3 and
  IndexTTS2. Primary scorer.
- EmpathicInsight-Voice (LAION, optional, `--eiv`): a Whisper-small encoder plus one small MLP head per dimension
  (40 emotions, valence, arousal, ...). The heads ship without inference code; ttsdata.emotion_eiv rebuilds the MLP
  from the checkpoints. Trained on synthetic speech, so its value on this TV mix is exactly what the calibration
  set measures.

A clip gets a coarse label only when the top class is one of the four coarse classes, confident
(emotion_min_conf) and clear of the runner-up coarse class (emotion_min_margin), and, when both scorers ran, when
both agree. Fine classes (fear, disgust, surprise) are never labelled: every published fine-tune found them
unreliable. Output per root: emotions.jsonl {id, e2v, eiv, coarse, conf}.

Calibration: `--calibration N` exports N clips, stratified over the predicted class (including unlabelled) and
the series, with a ratings.csv for native listeners; `--ratings FILE` scores the ratings (accuracy, per-class
precision / recall, confusion) and suggests per-class confidence thresholds that reach a target precision.
"""
from __future__ import annotations

import csv
import io
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from .config import Config

COARSE = ("angry", "happy", "sad", "neutral")
E2V_CLASSES = ("angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown")
RATING_CHOICES = ("angry", "happy", "sad", "neutral", "other", "unclear")


# ----------------------------------------------------------------------------------------------- pure logic
def coarse_label(scores: dict[str, float], min_conf: float, min_margin: float) -> tuple[str | None, float]:
    """(label or None, confidence of the top class). Only the four coarse classes can be labelled."""
    top = max(scores, key=scores.get)
    conf = float(scores[top])
    if top not in COARSE or conf < min_conf:
        return None, conf
    runner = max((scores[c] for c in COARSE if c != top), default=0.0)
    if conf - runner < min_margin:
        return None, conf
    return top, conf


def agree(first: str | None, second: str | None | str) -> str | None:
    """Label when both scorers agree; "missing" for a scorer that did not run lets the other decide."""
    if second == "missing":
        return first
    return first if first is not None and first == second else None


def calibration_sample(rows: list[dict], n: int, seed: int = 0) -> list[str]:
    """n clip ids, balanced over the predicted coarse class (None included) and, within a class, over the series.

    rows: {"id", "series", "coarse", "conf"}. Deterministic for a given seed."""
    rng = random.Random(seed)
    by_class: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_class[r.get("coarse")][str(r.get("series"))].append(r["id"])
    for series_map in by_class.values():
        for ids in series_map.values():
            ids.sort()
            rng.shuffle(ids)
    classes = sorted(by_class, key=lambda c: (c is None, str(c)))
    if not classes:
        return []
    quota = {c: n // len(classes) for c in classes}
    for c in classes[: n - sum(quota.values())]:
        quota[c] += 1
    picked: list[str] = []
    for c in classes:
        lists = [by_class[c][s] for s in sorted(by_class[c])]
        taken, k = 0, 0
        while taken < quota[c] and any(lists):
            lst = lists[k % len(lists)]
            if lst:
                picked.append(lst.pop())
                taken += 1
            k += 1
    picked = picked[:n]
    rng.shuffle(picked)  # interleave the classes: a block of one predicted class would bias the rater
    return picked


def parse_ratings(text: str) -> dict[str, str]:
    """ratings.csv -> {id: rating}; rows without a rating are skipped."""
    out = {}
    for row in csv.DictReader(io.StringIO(text)):
        rating = (row.get("rating") or "").strip().lower()
        if rating:
            out[row["id"].strip()] = rating
    return out


def score_ratings(preds: dict[str, tuple[str | None, float]], ratings: dict[str, str]) -> dict:
    """Accuracy over labelled clips, per-class precision / recall, confusion, and how often the model abstained."""
    per: dict = {c: {"predicted": 0, "correct": 0, "rated": 0} for c in COARSE}
    confusion: Counter = Counter()
    n = correct = labelled = 0
    for cid, rating in ratings.items():
        if cid not in preds:
            continue
        n += 1
        label, _ = preds[cid]
        if rating in per:
            per[rating]["rated"] += 1
        if label is not None:
            labelled += 1
            per[label]["predicted"] += 1
            confusion[(label, rating)] += 1
            if label == rating:
                correct += 1
                per[label]["correct"] += 1
    for c, d in per.items():
        d["precision"] = round(d["correct"] / d["predicted"], 4) if d["predicted"] else None
        d["recall"] = round(d["correct"] / d["rated"], 4) if d["rated"] else None
    return {"n": n, "labelled": labelled, "accuracy": round(correct / labelled, 4) if labelled else None,
            "abstained": n - labelled, "per_class": per, "confusion": dict(confusion)}


def suggest_thresholds(preds: dict[str, tuple[str | None, float]], ratings: dict[str, str],
                       target_precision: float = 0.76) -> dict[str, float | None]:
    """Per predicted class, the lowest confidence at which precision against the ratings reaches the target."""
    out: dict[str, float | None] = {}
    by_class: dict = defaultdict(list)
    for cid, (label, conf) in preds.items():
        if label is not None and cid in ratings:
            by_class[label].append((conf, ratings[cid] == label))
    for label, items in by_class.items():
        best = None
        for conf, _ in sorted(items):
            kept = [ok for c, ok in items if c >= conf]
            if kept and sum(kept) / len(kept) >= target_precision:
                best = conf
                break
        out[label] = best
    return out


# ----------------------------------------------------------------------------------------------- scorers
def _device(choice: str) -> str:
    if choice != "auto":
        return choice
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"  # FunASR has no MPS path


class E2V:
    def __init__(self, model: str, device: str):
        from funasr import AutoModel

        self.model = AutoModel(model=model, hub="ms", device=device, disable_update=True, disable_pbar=True, log_level="ERROR")

    def score(self, wav: Path) -> dict[str, float]:
        r = self.model.generate(str(wav), granularity="utterance", extract_embedding=False)[0]
        return parse_e2v(r["labels"], r["scores"])


def parse_e2v(labels: list[str], scores: list[float]) -> dict[str, float]:
    """FunASR labels come as "生气/angry" and "<unk>": keep the English name, call <unk> "unknown"."""
    out = {}
    for label, s in zip(labels, scores):
        name = label.split("/")[-1].strip().lower()
        out["unknown" if name in ("<unk>", "unk") else name] = round(float(s), 4)
    return out


# ----------------------------------------------------------------------------------------------- driver
def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def _roots(cfg: Config) -> list[tuple[str, Path]]:
    roots = [("dataset", cfg.dataset_dir)]
    if (cfg.root / "dataset_long" / "metadata.jsonl").exists():
        roots.append(("dataset_long", cfg.root / "dataset_long"))
    return roots


def score_roots(cfg: Config, limit: int = 0, eiv: bool = False, require_agreement: bool = False,
                eiv_ids: set[str] | None = None) -> dict[str, list[dict]]:
    """Score every clip of every root (cached per clip + duration + model); write emotions.jsonl per root.

    eiv: also score EmpathicInsight-Voice, for every clip (eiv_ids None) or only for eiv_ids (the calibration set).
    The coarse label comes from emotion2vec+; with require_agreement both scorers must agree."""
    from tqdm import tqdm

    device = _device(cfg.emotion_device)
    scorers: dict = {}
    results: dict[str, list[dict]] = {}
    for name, root in _roots(cfg):
        rows = _load_jsonl(root / "metadata.jsonl")
        if limit:
            rows = rows[:limit]
        cache_file = root / "emotions_cache.jsonl"
        cache: dict[str, dict] = {}
        for c in _load_jsonl(cache_file):
            cache[c["key"]] = c

        def key(r, which):
            return f"{r['id']}|{r['duration']}|{which}"

        def wants_eiv(r):
            return eiv and (eiv_ids is None or r["id"] in eiv_ids)

        todo = [r for r in rows if key(r, cfg.emotion_model) not in cache or (wants_eiv(r) and key(r, "eiv") not in cache)]
        if todo:
            if "e2v" not in scorers:
                scorers["e2v"] = E2V(cfg.emotion_model, device)
                if eiv:
                    from .emotion_eiv import EIV

                    scorers["eiv"] = EIV(device)
                print(f"[emotions] {cfg.emotion_model}{' + EmpathicInsight-Voice' if eiv else ''} on {device}")
            with cache_file.open("a", encoding="utf-8") as f, tqdm(total=len(todo), desc=name, unit="clip") as bar:
                for r in todo:
                    wav = root / "wavs" / f"{r['id']}.wav"
                    if key(r, cfg.emotion_model) not in cache:
                        c = {"key": key(r, cfg.emotion_model), "scores": scorers["e2v"].score(wav)}
                        cache[c["key"]] = c
                        f.write(json.dumps(c, ensure_ascii=False) + "\n")
                    if wants_eiv(r) and key(r, "eiv") not in cache:
                        c = {"key": key(r, "eiv"), "scores": scorers["eiv"].score(wav)}
                        cache[c["key"]] = c
                        f.write(json.dumps(c, ensure_ascii=False) + "\n")
                    f.flush()
                    bar.update(1)
        out = []
        for r in rows:
            e2v = cache[key(r, cfg.emotion_model)]["scores"]
            label, conf = coarse_label(e2v, cfg.emotion_min_conf, cfg.emotion_min_margin)
            row = {"id": r["id"], "series": r["series"], "e2v": e2v, "e2v_label": label, "conf": conf}
            if key(r, "eiv") in cache:
                from .emotion_eiv import coarse_from_eiv

                row["eiv"] = cache[key(r, "eiv")]["scores"]
                row["eiv_label"] = coarse_from_eiv(row["eiv"])
                row["coarse"] = agree(label, row["eiv_label"]) if require_agreement else label
            else:
                row["eiv"], row["eiv_label"] = None, None
                row["coarse"] = label
            out.append(row)
        with (root / "emotions.jsonl").open("w", encoding="utf-8") as f:
            for row in out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        results[name] = out
    return results


def report(results: dict[str, list[dict]], cfg: Config) -> str:
    lines = ["# Emotion labels (`ttsdata emotions`)", "",
             f"Scorer: {cfg.emotion_model}; a clip is labelled when the top class is coarse, confidence >= "
             f"{cfg.emotion_min_conf} and margin >= {cfg.emotion_min_margin}. Labels are automatic and uncalibrated "
             "until `--ratings` has been scored.", ""]
    for name, rows in results.items():
        n = len(rows)
        lab = Counter(r["coarse"] for r in rows)
        lines += [f"## `{name}/` ({n} clips)", "", "| coarse label | clips | share |", "|---|---|---|"]
        lines += [f"| {k if k else 'unlabelled'} | {v} | {v / n:.1%} |" for k, v in lab.most_common()]
        top = Counter(max(r["e2v"], key=r["e2v"].get) for r in rows)
        lines += ["", "Top emotion2vec+ class before the confidence rule: " + ", ".join(f"{k} {v}" for k, v in top.most_common()), ""]
        per_series: dict = defaultdict(Counter)
        for r in rows:
            per_series[r["series"]][r["coarse"] or "unlabelled"] += 1
        lines += ["| series | angry | happy | sad | neutral | unlabelled |", "|---|---|---|---|---|---|"]
        lines += [f"| {s} | " + " | ".join(str(c.get(k, 0)) for k in ("angry", "happy", "sad", "neutral", "unlabelled")) + " |"
                  for s, c in sorted(per_series.items())]
        lines.append("")
    return "\n".join(lines)


def export_calibration(cfg: Config, results: dict[str, list[dict]], n: int, seed: int = 0) -> Path:
    """Clips for native listeners: wavs (symlinks), ratings.csv to fill in, instructions."""
    out = cfg.dataset_dir / "calibrate_emotions"
    out.mkdir(exist_ok=True)
    for p in out.iterdir():
        if p.is_symlink():
            p.unlink()
    rows = [r for rs in results.values() for r in rs]
    root_of = {r["id"]: name for name, rs in results.items() for r in rs}
    texts = {}
    for name, root in _roots(cfg):
        for r in _load_jsonl(root / "metadata.jsonl"):
            texts[r["id"]] = r["text"]
    picked = calibration_sample(rows, n, seed)
    by_id = {r["id"]: r for r in rows}
    with (out / "ratings.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "file", "predicted", "conf", "text", "rating"])
        for k, cid in enumerate(picked):
            r = by_id[cid]
            name = f"{k:03d}_{cid}.wav"
            src = cfg.root / root_of[cid] / "wavs" / f"{cid}.wav"
            os.symlink(os.path.relpath(src, out), out / name)
            w.writerow([cid, name, r["coarse"] or "", f"{r['conf']:.2f}", texts.get(cid, ""), ""])
    (out / "README.md").write_text("\n".join([
        "# Emotion calibration set", "",
        f"{len(picked)} clips, balanced over the model's predicted class (including unlabelled) and the series.",
        "Listen to each file and fill the `rating` column of `ratings.csv` with one of: "
        + ", ".join(RATING_CHOICES) + ".",
        "Rate the delivery of the voice, not the meaning of the words. `other` = a clear emotion that is none of the "
        "four (fear, disgust, surprise...); `unclear` = cannot tell. Do not look at the `predicted` column first.", "",
        "Then: `ttsdata emotions --ratings data/dataset/calibrate_emotions/ratings.csv`", ""]), encoding="utf-8")
    return out


def score_calibration(cfg: Config, ratings_file: Path, target_precision: float = 0.76) -> dict:
    """Grade every scorer (emotion2vec+, EmpathicInsight-Voice where scored, their agreement) against the ratings."""
    rows = [r for name, root in _roots(cfg) for r in _load_jsonl(root / "emotions.jsonl")]
    ratings = parse_ratings(ratings_file.read_text(encoding="utf-8"))
    scorers = {"emotion2vec+": {r["id"]: (r["e2v_label"], r["conf"]) for r in rows}}
    if any(r.get("eiv") for r in rows):
        scorers["EmpathicInsight-Voice"] = {r["id"]: (r["eiv_label"], r["conf"]) for r in rows if r.get("eiv")}
        scorers["agreement"] = {r["id"]: (agree(r["e2v_label"], r["eiv_label"]), r["conf"]) for r in rows if r.get("eiv")}
    out = {}
    lines = ["# Emotion calibration", "", f"{len(ratings)} rated clips. Target precision {target_precision} "
             "(what a Turkish-specific classifier reached on TurEV-DB).", ""]
    for name, preds in scorers.items():
        m = score_ratings(preds, ratings)
        thr = suggest_thresholds(preds, ratings, target_precision) if name == "emotion2vec+" else {}
        out[name] = {"metrics": m, "thresholds": thr}
        lines += [f"## {name}", "", f"{m['n']} rated clips scored; labelled {m['labelled']}, accuracy on those "
                  f"{m['accuracy']}; abstained on {m['abstained']}.", "",
                  "| class | predicted | correct | precision | rated | recall |" + (" confidence for the target |" if thr else ""),
                  "|---|---|---|---|---|---|" + ("---|" if thr else "")]
        for c, d in m["per_class"].items():
            lines.append(f"| {c} | {d['predicted']} | {d['correct']} | {d['precision']} | {d['rated']} | {d['recall']} |"
                         + (f" {thr.get(c)} |" if thr else ""))
        lines += ["", "Confusion (predicted -> rated): " + ", ".join(f"{p}->{r} {n}" for (p, r), n in sorted(m["confusion"].items())), ""]
    lines += ["Use: `--auto-conf angry=X,sad=Y,happy=Z` on the manifest with the suggested emotion2vec+ confidences; drop a "
              "class whose precision never reaches the target (`None`). If agreement scores better than emotion2vec+ alone, "
              "run `ttsdata emotions --eiv-all --require-agreement`.", ""]
    text = "\n".join(lines)
    (cfg.dataset_dir / "EMOTION_CALIBRATION.md").write_text(text, encoding="utf-8")
    print(text)
    return out


def run(cfg: Config, limit: int = 0, calibration: int = 0, ratings: str | None = None, eiv: bool = False,
        eiv_all: bool = False, require_agreement: bool = False) -> dict:
    if ratings:
        return score_calibration(cfg, Path(ratings))
    results = score_roots(cfg, limit, eiv=eiv_all, require_agreement=require_agreement)
    if calibration:
        rows = [r for rs in results.values() for r in rs]
        picked = set(calibration_sample(rows, calibration, seed=0))
        if eiv and not eiv_all:  # EmpathicInsight on the calibration clips only: cheap, and enough to grade it
            results = score_roots(cfg, limit, eiv=True, require_agreement=require_agreement, eiv_ids=picked)
        out = export_calibration(cfg, results, calibration)
        print(f"calibration set: {out}/ratings.csv ({calibration} clips)")
    text = report(results, cfg)
    (cfg.dataset_dir / "EMOTIONS.md").write_text(text, encoding="utf-8")
    print(text)
    return {name: dict(Counter(r["coarse"] for r in rows)) for name, rows in results.items()}
