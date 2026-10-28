import pytest
from ttsdata.manifest import (EXCLUDE_FLAGS, combine_units, manifest_row, pair_references, replay_text, select_tier,
                              validate_row)


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


def test_combine_units_prefers_merged_and_never_duplicates():
    singles = [R(0), R(1), R(2)]
    merged = {"id": "c0_x2", "audio_file": "wavs/c0_x2.wav", "duration": 7.5, "cer": 0.02, "speaker_id": "s_spk00",
              "speaker_cluster_size": 20, "speaker_hint": None, "normalized_text": "Merhaba. Merhaba.", "split": "train",
              "members": ["c0", "c1"]}
    q = {f"c{i}": Q(i) for i in range(3)}
    q["c0_x2"] = {"id": "c0_x2", "music_db": None, "trail_sil": 0.1, "text_flags": []}
    units = combine_units(singles, [merged], q)
    assert sorted(u["id"] for u in units) == ["c0_x2", "c2"]
    assert {u["id"]: u["root"] for u in units} == {"c0_x2": "dataset_long", "c2": "dataset"}
    bad = {**merged, "cer": 0.2}  # a merge the tier rejects: its members stay usable as singles
    assert sorted(u["id"] for u in combine_units(singles, [bad], q)) == ["c0", "c1", "c2"]
    assert sorted(u["id"] for u in combine_units(singles, [], q)) == ["c0", "c1", "c2"]


def test_sound_flags_gate_drops_noisy_clips():
    rows, q = [R(0)], {"c0": Q(0)}
    for flag in EXCLUDE_FLAGS:
        assert select_tier(rows, q, soundtags={"c0": {"flags": [flag]}}) == [], flag
    assert EXCLUDE_FLAGS == {"overlap", "foreign_speech", "unintelligible", "censored", "channel"}
    for flag in ("music", "sfx", "silence"):
        assert select_tier(rows, q, soundtags={"c0": {"flags": [flag]}}) != [], flag
    assert select_tier(rows, q, soundtags={"c0": {"flags": ["bg_laugh"]}}) != []           # stage 1: fine
    assert select_tier(rows, q, soundtags={"c0": {"flags": ["bg_laugh"]}}, stage=2) == []  # stage 2: a laugh track would mislabel [gülme]
    assert select_tier(rows, q, soundtags={}) != []       # no tags for the row: nothing to exclude
    assert select_tier(rows, q, soundtags=None) != []     # soundtags not run: gate passes


def test_manifest_row_resolves_the_reference_by_its_own_root():
    merged_ref = {**R(1), "id": "c1_x2", "duration": 6.0}
    d = manifest_row(R(0), merged_ref, audio_root="../dataset/wavs_clean", ref_audio_root="../dataset_long/wavs_clean")
    assert d["audio"] == "../dataset/wavs_clean/c0.wav" and d["ref_audio"] == "../dataset_long/wavs_clean/c1_x2.wav"
    assert d["ref_duration"] == 6.0 and validate_row(d) == []


def test_combine_events_replaces_the_base_with_its_extended_clip():
    from ttsdata.manifest import combine_events
    units = [{**R(0), "root": "dataset"}, {**R(1), "root": "dataset"}, {**R(2), "id": "c2_x2", "root": "dataset_long", "members": ["c2", "c3"]}]
    ev = lambda i, base, tag="laugh", evidence="laugh_syllables", verified=True, dur=6.0: {  # noqa: E731
        **R(i), "id": f"{base}_{tag}", "base": base, "tag": tag, "edge": "start", "duration": dur, "verified": verified,
        "detector": {"evidence": evidence if verified else None, "reason": None if verified else "unconfirmed"},
        "text_tagged": f"[gülme] Merhaba.", "normalized_text_tagged": f"[gülme] Merhaba.", "cer": 0.0}
    events = [ev(10, "c0"), ev(11, "c0", tag="cry", evidence="clean", dur=8.0), ev(12, "c2_x2"), ev(13, "c9")]
    unc = [ev(14, "c1", verified=False)]
    bad = [ev(15, "c1", dur=25.0)]
    q = {r["id"]: Q(0) for r in units + events + unc + bad}
    out = combine_events(units, events, q, evidence={"laugh_syllables", "tagger", "clean"}, include_unconfirmed=False)
    ids = {u["id"]: u for u in out}
    assert set(ids) == {"c0_laugh", "c1", "c2_x2_laugh"}        # c0 and c2_x2 replaced; c9 has no base here: skipped
    assert ids["c0_laugh"]["root"] == "dataset_events"          # the stronger evidence wins over the longer clean one
    assert combine_events(units, events, q, evidence={"clean"}, include_unconfirmed=False)[0]["id"] in ("c0_cry", "c1")
    assert {u["id"] for u in combine_events(units, unc, q, evidence={"clean"}, include_unconfirmed=False)} == {"c0", "c1", "c2_x2"}
    assert "c1_laugh" in {u["id"] for u in combine_events(units, unc, q, evidence={"clean"}, include_unconfirmed=True)}
    # an extended clip that fails the tier (25 s): the base stays
    assert {u["id"] for u in combine_events(units, bad, q, evidence={"clean"}, include_unconfirmed=False)} == {"c0", "c1", "c2_x2"}


def test_stage_filter_drops_strong_styles_and_events_in_stage_1():
    from ttsdata.manifest import stage_filter
    rows = [{**R(0), "root": "dataset"}, {**R(1), "root": "dataset"}, {**R(2), "id": "c2_laugh", "root": "dataset_events"}]
    tags = {"c0": {"style": ["crying"], "flags": [], "events": []}, "c1": {"style": ["dialect"], "flags": [], "events": []}}
    assert [r["id"] for r in stage_filter(rows, tags, stage=1)] == ["c1"]
    assert [r["id"] for r in stage_filter(rows, tags, stage=2)] == ["c0", "c1", "c2_laugh"]


def test_restore_bases_returns_the_base_when_the_tag_is_not_written():
    from ttsdata.manifest import restore_bases
    units = [{**R(0), "root": "dataset"}, {**R(1), "root": "dataset"}, {**R(2), "root": "dataset"}, {**R(3), "duration": 1.5, "root": "dataset"}]
    ev = lambda i, base, tag: {**R(i), "id": f"{base}_{tag}", "base": base, "tag": tag, "root": "dataset_events", "duration": 6.0}  # noqa: E731
    rows = [ev(10, "c0", "laugh"), units[1], ev(12, "c2", "cry"), ev(13, "c3", "cry")]
    q = {r["id"]: Q(0) for r in units + rows}
    out = restore_bases(rows, {"laugh"}, units, q)
    # laugh is written: its row stays; cry is not: c2 comes back in its place, c3 (1.5 s) fails the tier and is gone
    assert [r["id"] for r in out] == ["c0_laugh", "c1", "c2"] and out[2]["root"] == "dataset"
    assert [r["id"] for r in restore_bases(rows, {"laugh", "cry"}, units, q)] == ["c0_laugh", "c1", "c2_cry", "c3_cry"]


def _rows(n, series="A", dur=3600.0, start=0, **extra):
    return [{"id": f"{series}{start + i:04d}", "series": series, "duration": dur, **extra} for i in range(n)]


def test_cap_hours_without_taken_keeps_the_series_cap():
    from ttsdata.manifest import cap_hours
    rows = _rows(50, "A") + _rows(10, "B")
    out = cap_hours(rows, max_hours=20, max_series_share=0.6, seed=0)
    hours = {s: sum(r["duration"] for r in out if r["series"] == s) / 3600 for s in "AB"}
    assert sum(hours.values()) == pytest.approx(20, abs=2)
    assert hours["A"] <= 0.6 * 20 + 1
    assert cap_hours(rows[:5], max_hours=20, max_series_share=0.6, seed=0) == rows[:5]  # under the cap: all kept


def test_cap_hours_keeps_taken_rows_and_counts_them_against_the_caps():
    from ttsdata.manifest import cap_hours
    rows = _rows(50, "A") + _rows(50, "B")
    taken = [r for r in rows if r["series"] == "A"][:8]
    out = cap_hours(rows, max_hours=20, max_series_share=0.5, seed=0, taken=taken)
    ids = [r["id"] for r in out]
    assert len(ids) == len(set(ids))
    assert {r["id"] for r in taken} <= set(ids)
    hours = {s: sum(r["duration"] for r in out if r["series"] == s) / 3600 for s in "AB"}
    assert sum(hours.values()) == pytest.approx(20, abs=2)
    assert hours["A"] <= 0.5 * 20 + 1  # the 8 taken hours count toward A's 10 h cap


def test_labels_first_takes_the_rarest_label_first_within_quota_and_budget():
    from ttsdata.manifest import labels_first
    rows = (_rows(3, "A", dur=60, label="cry") + _rows(40, "A", dur=60, start=100, label="happy")
            + _rows(40, "B", dur=60, label=None))
    lab = lambda r: [r["label"]] if r.get("label") else []  # noqa: E731
    out = labels_first(rows, lab, max_hours=10, per_label=20, seed=0)
    got = [r["label"] for r in out]
    assert got.count("cry") == 3 and got.count("happy") == 20 and None not in got
    assert out == labels_first(rows, lab, max_hours=10, per_label=20, seed=0)  # seeded
    # budget of 0.1 h = 6 one-minute clips: the rare label still gets all its rows first
    small = labels_first(rows, lab, max_hours=0.1, per_label=20, seed=0)
    assert [r["label"] for r in small].count("cry") == 3 and len(small) == 6
    # a row carrying two labels counts toward both quotas
    both = [{"id": "x", "series": "A", "duration": 60, "labs": ["cry", "laugh"]}]
    assert len(labels_first(both, lambda r: r["labs"], max_hours=1, per_label=1, seed=0)) == 1
