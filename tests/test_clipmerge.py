from pathlib import Path

import numpy as np
import soundfile as sf

from ttsdata.clipmerge import (_cache_key, _compatible, _cut_episode, _verify, build_chains, join_texts, listen_picks,
                               merged_row, output_dir, seg_index, summarize, superseded)
from ttsdata.config import Config

RNG = np.random.default_rng(0)
DIM = 16


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


VOICE = unit(RNG.normal(size=DIM))
OTHER = unit(RNG.normal(size=DIM))


def C(i, s, e, uid="ep", spk="ep_spk00", hint=None, split="train", text=None, flags=(), size=50):
    return {"id": f"{uid}_{i:05d}", "uid": uid, "orig_start": s, "orig_end": e, "duration": round(e - s, 3),
            "speaker_id": spk, "speaker_cluster_size": size, "speaker_hint": hint, "split": split,
            "text": text or f"Cümle {i}.", "normalized_text": text or f"Cümle {i}.", "flags": list(flags),
            "series": "S", "source": "trt", "episode": 1, "sample_rate": 24000}


def vecs(rows, overrides=None):
    out = {r["id"]: VOICE for r in rows}
    out.update(overrides or {})
    return out


def ids(chains):
    return [[seg_index(r["id"]) for r in c] for c in chains]


def test_seg_index_handles_underscored_uids():
    assert seg_index("esrefruya_ep001_00012") == 12
    assert seg_index("5Q4eUJ7d5RM_00072") == 72


def test_build_chains_joins_consecutive_close_same_voice_cues():
    rows = [C(0, 0.0, 1.5), C(1, 1.8, 3.0), C(2, 3.2, 4.5), C(3, 9.0, 11.0)]
    assert ids(build_chains(rows, vecs(rows), max_gap=0.7, min_cos=0.4, max_dur=15.0)) == [[0, 1, 2]]


def test_build_chains_breaks_on_each_rule():
    def chains(rows, v=None, **kw):
        return ids(build_chains(rows, v or vecs(rows), **{"max_gap": 0.7, "min_cos": 0.4, "max_dur": 15.0, **kw}))
    assert chains([C(0, 0, 1), C(1, 1.8, 3)]) == []                       # gap 0.8 s
    assert chains([C(0, 0, 1), C(2, 1.2, 3)]) == []                       # a cue in between was dropped
    rows = [C(0, 0, 1), C(1, 1.2, 3)]
    assert chains(rows, vecs(rows, {rows[1]["id"]: OTHER})) == []         # another voice
    assert chains([C(0, 0, 1), C(1, 1.2, 3, split="val")]) == []          # never across train / val
    assert chains([C(0, 0, 1, hint="Nisan"), C(1, 1.2, 3, hint="Faruk")]) == []
    assert chains([C(0, 0, 1, hint="Nisan"), C(1, 1.2, 3, hint=None)]) == [[0, 1]]
    assert chains([C(0, 0, 1, hint="Nisan ses"), C(1, 1.2, 3, hint="Nisan")]) == [[0, 1]]
    assert chains([C(0, 0, 1), C(1, 1.2, 3, uid="other")]) == []          # episodes never mix
    assert chains([C(0, 0, 6), C(1, 6.2, 12), C(2, 12.2, 16)]) == [[0, 1]]  # 16 s > max_dur: C2 starts anew, alone
    assert chains([C(0, 0, 1), C(1, 0.8, 3)]) == [[0, 1]]                 # slightly overlapping edges are fine


def test_build_chains_checks_the_chain_centroid_not_only_neighbours():
    a, c = np.eye(3)[0], np.eye(3)[1]
    b = unit([0.5, 0.5, np.sqrt(0.5)])  # cos(a,b) = cos(b,c) = 0.5, cos(a,c) = 0
    rows = [C(0, 0, 1), C(1, 1.2, 2), C(2, 2.2, 3)]
    v = {rows[0]["id"]: a, rows[1]["id"]: b, rows[2]["id"]: c}
    assert ids(build_chains(rows, v, max_gap=0.7, min_cos=0.4, max_dur=15.0)) == [[0, 1]]


def test_join_texts_keeps_member_offsets():
    text, ends = join_texts(["Belki ölümden korkmadığı için...", "belki herkese göstermek için."])
    assert text == "Belki ölümden korkmadığı için... belki herkese göstermek için."
    assert ends == [len("Belki ölümden korkmadığı için..."), len(text)]


def test_merged_row_fields():
    chain = [C(4, 10.0, 11.5, spk="a", hint=None, flags=["fragment_start"], text="Bak."),
             C(5, 11.8, 13.0, spk="b", hint="Nisan", flags=["fragment_end", "fragment_start"], text="Sana söylüyorum..."),
             C(6, 13.2, 14.5, spk="a", hint=None, flags=["fragment_start", "fragment_end"], text="dinle beni...")]
    m = merged_row(chain)
    assert m["id"] == "ep_00004_x3" and m["members"] == ["ep_00004", "ep_00005", "ep_00006"]
    assert m["text"] == "Bak. Sana söylüyorum... dinle beni..." and m["member_text_ends"] == [4, 23, 37]
    assert m["orig_start"] == 10.0 and m["orig_end"] == 14.5 and m["duration"] == 4.5
    assert m["member_spans"] == [[10.0, 11.5], [11.8, 13.0], [13.2, 14.5]]
    assert m["speaker_id"] == "a" and m["speaker_hint"] == "Nisan" and m["split"] == "train"
    assert m["flags"] == ["fragment_end", "fragment_start", "merged"]  # outer edges only, plus the marker
    assert m["audio_file"] == "wavs/ep_00004_x3.wav"


def test_superseded_lists_members_of_kept_merges():
    merged = [{"id": "m1", "members": ["a", "b"]}, {"id": "m2", "members": ["c", "d", "e"]}]
    assert superseded(merged) == {"a", "b", "c", "d", "e"}


def test_listen_picks_mixes_random_and_weakest_links():
    kept = [{"id": f"m{i}", "members": ["a", "b"]} for i in range(10)]
    weakest = {f"m{i}": 0.4 + 0.05 * i for i in range(10)}
    picks = listen_picks(kept, weakest, n=3, seed=0)
    weak = [cid for group, cid in picks if group == "weakest_link"]
    assert weak == ["m0", "m1", "m2"]
    rand = [cid for group, cid in picks if group == "random"]
    assert len(rand) == 3 and not set(rand) & set(weak)


def test_summarize_reports_tier_level_gain():
    rows = [C(0, 0, 1.5), C(1, 1.7, 3.2), C(2, 10, 14)]
    rows[2]["cer"] = 0.0
    for r in rows[:2]:
        r["cer"] = 0.0
    m = merged_row(rows[:2])
    m["cer"] = 0.04
    st = summarize(rows, [m], [m], __import__("collections").Counter(), Config())
    assert st["clips_ge_3s_singles_only"] == 1 and st["clips_ge_3s_with_merges"] == 2
    assert st["tier_hours_singles_only"] == round(4 / 3600, 3)              # cer <= 0.05 and >= 3 s
    assert st["tier_hours_with_merges"] == round((4 + 3.2) / 3600, 3)
    m["cer"] = 0.08                                                        # kept by the merger, not by the tier
    st = summarize(rows, [m], [m], __import__("collections").Counter(), Config())
    assert st["tier_hours_with_merges"] == round(4 / 3600, 3)


def _pair(cos):
    a = np.eye(DIM)[0]
    b = cos * np.eye(DIM)[0] + np.sqrt(1 - cos ** 2) * np.eye(DIM)[1]
    return a, b


def test_build_chains_is_stricter_across_clusters_and_after_questions():
    kw = {"max_gap": 0.7, "min_cos": 0.4, "max_dur": 15.0, "cross_cluster_cos": 0.5, "after_question_cos": 0.55}

    def chains(cos, spk2="ep_spk00", text1="Gel."):
        rows = [C(0, 0, 1, text=text1), C(1, 1.2, 2.5, spk=spk2)]
        a, b = _pair(cos)
        return ids(build_chains(rows, {rows[0]["id"]: a, rows[1]["id"]: b}, **kw))
    assert chains(0.45) == [[0, 1]]                          # same cluster: 0.4 is enough
    assert chains(0.45, spk2="ep_spk07") == []              # another cluster: needs 0.5
    assert chains(0.52, spk2="ep_spk07") == [[0, 1]]
    assert chains(0.50, text1="Ne oldu?") == []             # a question is often answered by someone else
    assert chains(0.60, text1="Ne oldu?") == [[0, 1]]


def test_compatible_labels_compare_full_names():
    assert not _compatible([C(0, 0, 1, hint="Erkek 1")], C(1, 1.2, 3, hint="Erkek 2"))
    assert not _compatible([C(0, 0, 1, hint="Atıf'ın annesi")], C(1, 1.2, 3, hint="Atıf'ın babası"))
    assert not _compatible([C(0, 0, 1, hint="Mustafa")], C(1, 1.2, 3, hint="Mustafa Ali"))
    assert _compatible([C(0, 0, 1, hint="Nisan ses")], C(1, 1.2, 3, hint="Nisan"))
    assert _compatible([C(0, 0, 1, hint="Erkek")], C(1, 1.2, 3, hint=None))
    assert _compatible([C(0, 0, 1, hint="Ömer ağlayarak")], C(1, 1.2, 3, hint="Ömer"))


def test_merged_row_keeps_the_name_only_and_member_hints():
    chain = [C(0, 0.0, 1.0), C(1, 1.2, 2.5, hint="Ömer ağlayarak"), C(2, 2.7, 4.0)]
    m = merged_row(chain)
    assert m["speaker_hint"] == "Ömer" and m["member_hints"] == [None, "Ömer ağlayarak", None]


def test_output_dir_keeps_smoke_runs_apart():
    cfg = Config(root=Path("/tmp/x"))
    assert output_dir(cfg, 0) == Path("/tmp/x/dataset_long")
    assert output_dir(cfg, 2) == Path("/tmp/x/dataset_long_smoke")


def _raw_episode(tmp_path, seconds=10.0, sr=24000):
    d = tmp_path / "raw" / "trt" / "ep"
    d.mkdir(parents=True)
    t = np.arange(int(seconds * sr)) / sr
    sf.write(d / "audio.wav", (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)
    return Config(root=tmp_path)


def test_cut_episode_recuts_when_the_span_changes_or_the_wav_is_partial(tmp_path):
    cfg = _raw_episode(tmp_path)
    wav_dir = tmp_path / "out" / "wavs"
    wav_dir.mkdir(parents=True)
    index = {}
    cand = {"id": "ep_00003_x2", "uid": "ep", "source": "trt", "orig_start": 1.0, "orig_end": 4.0}
    _cut_episode(cfg, "ep", "trt", [cand], wav_dir, index)
    assert abs(sf.info(wav_dir / "ep_00003_x2.wav").duration - 3.0) < 0.01
    moved = {**cand, "orig_start": 1.4, "orig_end": 4.8}   # build re-run with new edges: same id, new span
    _cut_episode(cfg, "ep", "trt", [moved], wav_dir, index)
    assert abs(sf.info(wav_dir / "ep_00003_x2.wav").duration - 3.4) < 0.01
    sf.write(wav_dir / "ep_00003_x2.wav", np.zeros(2400, dtype=np.float32), 24000)  # a run killed mid-write
    index.clear()
    _cut_episode(cfg, "ep", "trt", [moved], wav_dir, index)
    assert abs(sf.info(wav_dir / "ep_00003_x2.wav").duration - 3.4) < 0.01
    assert not list(wav_dir.glob("*.tmp*"))


def test_cache_key_includes_span_text_and_model():
    row = {"id": "a_x2", "orig_start": 1.0, "orig_end": 4.0, "text": "Gel."}
    k = _cache_key(row, "mlx:whisper-large-v3-turbo")
    assert k != _cache_key({**row, "orig_end": 4.1}, "mlx:whisper-large-v3-turbo")
    assert k != _cache_key({**row, "text": "Git."}, "mlx:whisper-large-v3-turbo")
    assert k != _cache_key(row, "faster:large-v3")


def test_verify_only_transcribes_uncached_candidates(tmp_path, monkeypatch):
    import ttsdata.asr as asr

    calls = []

    def fake_verify_pack(pack, cfg):
        calls.extend(r["id"] for r in pack)
        return [{"id": r["id"], "cer": 0.0, "trim_start": 0.0} for r in pack]
    monkeypatch.setattr(asr, "verify_pack", fake_verify_pack)
    cfg = Config(root=tmp_path)
    cands = [{"id": "a_x2", "orig_start": 0.0, "orig_end": 3.0, "duration": 3.0, "text": "A."},
             {"id": "b_x2", "orig_start": 5.0, "orig_end": 9.0, "duration": 4.0, "text": "B."}]
    cache = {_cache_key(cands[0], "m"): {"cer": 0.0}}
    cache_file = tmp_path / "asr_cache.jsonl"
    _verify(cfg, cands, tmp_path, cache, cache_file, model_tag="m")
    assert calls == ["b_x2"] and _cache_key(cands[1], "m") in cache
    assert len(cache_file.read_text().splitlines()) == 1
