import numpy as np

from ttsdata.asr import pack_records, split_pack_result


def test_pack_records_respects_budget_and_covers_all():
    recs = [{"id": str(i), "window_dur": 4.0 + (i % 3)} for i in range(30)]
    packs = pack_records(recs, pack_seconds=28.0, gap=1.0, seed=1)
    assert sorted(r["id"] for p in packs for r in p) == sorted(r["id"] for r in recs)
    for p in packs:
        assert sum(r["window_dur"] + 1.0 for r in p) <= 28.0 or len(p) == 1


def test_pack_records_single_long_clip():
    packs = pack_records([{"id": "a", "window_dur": 40.0}, {"id": "b", "window_dur": 3.0}], 28.0, 1.0, 1)
    assert len(packs) == 2 and all(len(p) == 1 for p in packs)


def test_split_pack_result_assigns_words_by_time():
    words = [
        {"word": "gel", "start": 0.2, "end": 0.5, "prob": 0.9},          # clip 0 (0-3)
        {"word": "hallu", "start": 3.4, "end": 3.6, "prob": 0.1},        # gap (3-4): nearest clip 0 at 0.5 s > slack 0.3 -> dropped
        {"word": "evet", "start": 4.1, "end": 4.4, "prob": 0.9},         # clip 1 (4-6)
        {"word": "tamam", "start": 5.95, "end": 6.2, "prob": 0.9},       # straddles clip 1 end, midpoint 6.07 within slack
    ]
    segs = [{"start": 0.0, "end": 3.0, "no_speech_prob": 0.01, "avg_logprob": -0.2},
            {"start": 4.0, "end": 6.5, "no_speech_prob": 0.05, "avg_logprob": -0.4}]
    parts = split_pack_result({"words": words, "segments": segs}, offsets=[0.0, 4.0], durations=[3.0, 2.0], slack=0.3)
    assert [w["word"] for w in parts[0]["words"]] == ["gel"]
    assert [w["word"] for w in parts[1]["words"]] == ["evet", "tamam"]
    assert abs(parts[1]["words"][0]["start"] - 0.1) < 1e-9 and parts[1]["words"][1]["end"] == 2.0  # clipped to the clip
    assert parts[0]["no_speech_prob"] == 0.01 and abs(parts[1]["no_speech_prob"] - 0.05) < 1e-9


def test_split_pack_result_empty_clip():
    parts = split_pack_result({"words": [], "segments": []}, offsets=[0.0], durations=[2.0], slack=0.3)
    assert parts[0]["words"] == [] and parts[0]["no_speech_prob"] == 1.0
