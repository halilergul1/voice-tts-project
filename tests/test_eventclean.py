from ttsdata.eventclean import flagged_ids, narrator_overlap, pitch_conflict, who_conflict


def test_narrator_overlap():
    ev = {"start": 100.0, "end": 102.0}
    assert narrator_overlap(ev, [(101.5, 104.0)])
    assert not narrator_overlap(ev, [(101.9, 104.0)])       # 0.1 s: a breath of overlap, not speech over the cry
    assert not narrator_overlap(ev, [])


def test_who_conflict():
    cl = {("S", "Afra"): "S_spk3"}
    assert who_conflict("Afra", "S", "S_spk7", cl)
    assert not who_conflict("Afra", "S", "S_spk3", cl)
    assert not who_conflict("Kerem", "S", "S_spk7", cl)      # name not mapped: no evidence either way
    assert not who_conflict(None, "S", "S_spk7", cl)


def test_pitch_conflict():
    assert pitch_conflict(300.0, "male") and pitch_conflict(120.0, "female")
    assert not pitch_conflict(230.0, "male")                 # a man crying can sound high
    assert not pitch_conflict(None, "male") and not pitch_conflict(300.0, None)


def test_flagged_ids_takes_rows_with_any_check_true():
    rows = [{"id": "a", "narrator": False, "who_conflict": False, "pitch_conflict": False},
            {"id": "b", "narrator": True, "who_conflict": False, "pitch_conflict": False},
            {"id": "c", "narrator": False, "who_conflict": False, "pitch_conflict": True}]
    assert flagged_ids(rows) == {"b", "c"}
