from ttsdata.manifest import manifest_row, pair_references, replay_text, select_tier, validate_row


def R(i, dur=4.0, cer=0.0, spk="s_spk00", size=20, hint=None, text="Merhaba."):
    return {"id": f"c{i}", "audio_file": f"wavs/c{i}.wav", "duration": dur, "cer": cer, "speaker_id": spk,
            "speaker_cluster_size": size, "speaker_hint": hint, "normalized_text": text, "split": "train"}


def Q(i, music=-40.0, trail=0.1, flags=()):
    return {"id": f"c{i}", "music_db": music, "trail_sil": trail, "text_flags": list(flags)}


def test_select_tier_applies_every_gate():
    rows = [R(0), R(1, dur=2.9), R(2, cer=0.06), R(3, size=5), R(4, text="Saat 3.")]
    q = {f"c{i}": Q(i) for i in range(5)}
    q["c4"]["text_flags"] = ["digits"]
    assert [r["id"] for r in select_tier(rows, q)] == ["c0"]
    assert select_tier([R(0)], {"c0": Q(0, trail=0.8)}) == []
    assert select_tier([R(0)], {"c0": Q(0, music=-10.0)}) == []
    assert select_tier([R(0, dur=3.0)], {"c0": Q(0)}) != []  # 3.0 s is in, 2.9 s is out
    assert select_tier([R(0, dur=20.5)], {"c0": Q(0)}) == []
    assert select_tier([R(0)], {}) == []  # no quality row: not audited, not eligible
    assert select_tier([R(0)], {"c0": Q(0, music=None)}) != []  # music not measured (no separation run): allowed
    # a clip whose music was stripped by separation is rescued, unless the music was louder than the voice
    assert select_tier([R(0)], {"c0": {**Q(0, music=-10.0), "source": "separated"}}) != []
    assert select_tier([R(0)], {"c0": {**Q(0, music=-10.0), "source": "original"}}) == []
    assert select_tier([R(0)], {"c0": {**Q(0, music=2.0), "source": "separated"}}) == []


def test_pair_references_same_cluster_never_self():
    rows = [R(i, spk="a", size=10) for i in range(10)] + [R(i + 10, spk="b", size=10) for i in range(10)]
    pairs = pair_references(rows, ratio=0.5, seed=1)
    by = {r["id"]: r for r in rows}
    assert 8 <= len(pairs) <= 12
    for t, ref in pairs.items():
        assert t != ref and by[t]["speaker_id"] == by[ref]["speaker_id"]


def test_pair_references_prefers_same_hint_and_skips_singletons():
    rows = [R(0, hint="Ali"), R(1, hint="Ali"), R(2, hint=None), R(3, hint="Veli"), R(4, spk="solo")]
    pairs = pair_references(rows, ratio=1.0, seed=0)
    assert pairs["c0"] == "c1" and pairs["c1"] == "c0"
    assert pairs["c3"] == "c2"  # no other Veli: an unlabelled clip beats a clip labelled with another name
    assert "c4" not in pairs
    assert pair_references(rows, ratio=1.0, seed=3) == pair_references(rows, ratio=1.0, seed=3)  # deterministic


def test_manifest_row_and_validate():
    d = manifest_row(R(0), R(1), audio_root="../dataset/wavs_clean")
    assert d["audio"] == "../dataset/wavs_clean/c0.wav" and d["ref_audio"] == "../dataset/wavs_clean/c1.wav"
    assert d["text"] == "Merhaba." and d["duration"] == 4.0 and d["ref_duration"] == 4.0 and d["dataset_id"] == 0
    assert validate_row(d) == []
    solo = manifest_row(R(0), None, audio_root="../dataset/wavs_clean", dataset_id=1)
    assert "ref_audio" not in solo and solo["dataset_id"] == 1 and validate_row(solo) == []
    assert "ref_duration" in " ".join(validate_row({**d, "ref_duration": None}))
    assert "digits" in " ".join(validate_row({**d, "text": "Saat 3."}))
    assert "duration" in " ".join(validate_row({**d, "duration": 0.5}))
    assert "text" in " ".join(validate_row({**d, "text": "   "}))


def test_select_tier_fails_closed_on_incomplete_quality_rows():
    assert select_tier([R(0)], {"c0": {"id": "c0", "music_db": -30.0}}) == []  # separation ran, audit did not


def test_replay_text_normalises_or_rejects():
    assert replay_text("Toplantı 21 Mart'ta yapılacak.") == "Toplantı yirmi bir Mart'ta yapılacak."
    assert replay_text("Merhaba dünya.") == "Merhaba dünya."
    assert replay_text("(Gülüşmeler) Merhaba.") is None
    assert replay_text("   ") is None
