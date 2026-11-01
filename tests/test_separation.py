import numpy as np

from ttsdata.separation import choose_source, relative_loudness_db


def test_relative_loudness_db_is_instrumental_minus_vocals():
    v = np.full(24000, 0.1, dtype=np.float32)
    assert abs(relative_loudness_db(v, v * 0.1) - (-20.0)) < 0.1
    assert relative_loudness_db(v, np.zeros_like(v)) <= -80.0
    assert relative_loudness_db(np.zeros_like(v), np.zeros_like(v)) <= -80.0  # both silent: treat as no music


def test_choose_source_threshold():
    # the separator leaves a -12..-25 dB "instrumental" residue even on clean read speech, so the default sits above that
    assert choose_source(-35.0) == "original"
    assert choose_source(-12.0) == "original"
    assert choose_source(-8.0) == "separated"
    assert choose_source(-10.0) == "separated"
    assert choose_source(-10.0, threshold_db=-5.0) == "original"


def test_choose_source_keeps_original_when_the_stem_is_unreliable():
    assert choose_source(5.0) == "original"  # voice ended up in the instrumental stem
    assert choose_source(-5.0, vocal_ratio=0.1) == "original"  # vocal stem lost most of the energy
    assert choose_source(-5.0, vocal_ratio=0.8) == "separated"


def test_manifest_ids_picks_the_clips_of_one_root_from_audio_and_ref_audio():
    from ttsdata.separation import manifest_ids
    rows = [{"audio": "../dataset/wavs/a.wav", "ref_audio": "../dataset_long/wavs/m1.wav"},
            {"audio": "../dataset_long/wavs_clean/m2.wav"},
            {"audio": "../dataset_events/wavs/e1_laugh.wav", "ref_audio": "../dataset/wavs/b.wav"},
            {"audio": "../fleurs_tr/123.wav", "dataset_id": 1}]
    assert manifest_ids(rows, "dataset") == {"a", "b"}
    assert manifest_ids(rows, "dataset_long") == {"m1", "m2"}
    assert manifest_ids(rows, "dataset_events") == {"e1_laugh"}
    assert manifest_ids(rows, "fleurs_tr") == set()  # replay clips sit outside a dataset root's wav folder


def test_event_base_ids_are_the_bases_of_the_event_clips_a_manifest_uses():
    from ttsdata.separation import event_base_ids
    rows = [{"audio": "../dataset_events/wavs/e1_laugh.wav"}, {"audio": "../dataset/wavs/a.wav"}]
    events = [{"id": "e1_laugh", "base": "b1"}, {"id": "e2_cry", "base": "b2"}, {"id": "e3_x"}]
    # the stage 2 builder needs a cleaned base next to each event clip (it replaces the base, or the base returns)
    assert event_base_ids(rows, events) == {"b1"}
    assert event_base_ids([{"audio": "../dataset_events/wavs/e3_x.wav"}], events) == set()


def test_shard_splits_ids_into_disjoint_covering_parts():
    from ttsdata.separation import shard
    ids = [f"c{i}" for i in range(10)]
    parts = [shard(ids, k, 3) for k in range(3)]
    assert sorted(sum(parts, [])) == sorted(ids)
    assert all(not set(parts[i]) & set(parts[j]) for i in range(3) for j in range(i + 1, 3))
    assert shard(ids, 0, 1) == ids
    assert shard(list(reversed(ids)), 1, 3) == parts[1]  # independent of input order


def test_merge_separation_folds_shard_rows_into_quality():
    from ttsdata.separation import merge_separation
    quality = {"a": {"id": "a", "trail_sil": 0.1, "music_db": None}, "b": {"id": "b", "trail_sil": 0.2}}
    rows = [{"id": "a", "music_db": -5.0, "source": "separated", "vocal_ratio": 0.9},
            {"id": "c", "music_db": -30.0, "source": "original", "vocal_ratio": 1.0}]
    out = merge_separation(quality, rows)
    assert out["a"] == {"id": "a", "trail_sil": 0.1, "music_db": -5.0, "source": "separated", "vocal_ratio": 0.9}
    assert out["b"] == {"id": "b", "trail_sil": 0.2}
    assert out["c"]["source"] == "original"


def test_stems_for_finds_both_stems_of_an_aliased_input():
    """The separator rewrites file names (EL__x -> EL_x), so inputs go in under numbered aliases."""
    from ttsdata.separation import alias, stems_for
    outs = ["/t/00001_(Instrumental)_model_bs_roformer.wav", "/t/00001_(Vocals)_model_bs_roformer.wav",
            "/t/00010_(Vocals)_model_bs_roformer.wav", "/t/00010_(Instrumental)_model_bs_roformer.wav"]
    assert alias(1) == "00001" and alias(10) == "00010"
    v, i = stems_for(outs, alias(1))
    assert v.name.startswith("00001_(Vocals)") and i.name.startswith("00001_(Instrumental)")
    assert stems_for(outs, alias(2)) == (None, None)


def test_ids_to_separate_redoes_kept_originals_once():
    from ttsdata.separation import ids_to_separate
    ids = ["a", "b", "c", "d"]
    done = {"a", "b", "c"}  # files in wavs_clean
    quality = {"a": {"source": "separated"}, "b": {"source": "original"}, "c": {"source": "original"}}
    assert ids_to_separate(ids, done, quality) == ["d"]
    # --redo-original: kept originals carry a background bed (median -18 dB vs -63 dB separated, 2026-10-29)
    assert ids_to_separate(ids, done, quality, redo_original=True) == ["b", "c", "d"]
    # a restart skips clips a shard file already holds from this run
    assert ids_to_separate(ids, done, quality, redo_original=True, in_shards={"b"}) == ["c", "d"]


def test_separate_batch_keeps_the_outputs_of_the_inputs_that_worked():
    from ttsdata.separation import separate_batch

    class PartialFailure(RuntimeError):  # shaped like audio-separator's BatchSeparationError
        def __init__(self, ok, failures):
            super().__init__("Separation failed for 1 input(s)")
            self.successful_files, self.failures = ok, failures

    class Sep:
        def __init__(self, exc=None):
            self.exc = exc

        def separate(self, paths):
            if self.exc:
                raise self.exc
            return [f"{p}_(Vocals).wav" for p in paths]

    assert separate_batch(Sep(), ["a", "b"]) == ["a_(Vocals).wav", "b_(Vocals).wav"]
    exc = PartialFailure(["a_(Vocals).wav", "a_(Instrumental).wav"], [("b", ValueError("empty output"))])
    assert separate_batch(Sep(exc), ["a", "b"]) == ["a_(Vocals).wav", "a_(Instrumental).wav"]
    import pytest
    with pytest.raises(OSError):  # any other error still stops the run
        separate_batch(Sep(OSError("disk full")), ["a"])


def test_manifest_clip_ids_reads_the_manifests_and_adds_event_bases_for_the_main_root(tmp_path):
    import json

    from ttsdata.separation import manifest_clip_ids
    (tmp_path / "dataset_events").mkdir()
    (tmp_path / "dataset_events" / "metadata.jsonl").write_text(
        json.dumps({"id": "e1_laugh", "base": "b1"}) + "\n" + json.dumps({"id": "e2_cry", "base": "b2"}) + "\n")
    train, val = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    train.write_text(json.dumps({"audio": "../dataset/wavs/a.wav", "ref_audio": "../dataset_long/wavs/m1.wav"}) + "\n"
                     + json.dumps({"audio": "../dataset_events/wavs/e1_laugh.wav"}) + "\n")
    val.write_text(json.dumps({"audio": "../dataset/wavs_sidon/c.wav"}) + "\n")
    # the clips separate_vocals.py --manifests cleans: the root's own clips, plus the bases of the event clips
    assert manifest_clip_ids(tmp_path / "dataset", [train, val]) == {"a", "b1", "c"}
    assert manifest_clip_ids(tmp_path / "dataset_long", [str(train)]) == {"m1"}  # event bases live in dataset only
