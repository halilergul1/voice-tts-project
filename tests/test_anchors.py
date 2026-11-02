from ttsdata.anchors import anchor_keep, pick_sentences


def test_pick_sentences_filters_and_excludes():
    rows = [{"normalized_text": f"Cümle {w}.", "duration": 4.0, "split": "train", "labelled": False} for w in ("bir", "iki", "üç")]
    rows += [{"normalized_text": "Kısa.", "duration": 2.0, "split": "train", "labelled": False},
             {"normalized_text": "Val.", "duration": 4.0, "split": "val", "labelled": False},
             {"normalized_text": "Öfkeli.", "duration": 4.0, "split": "train", "labelled": True}]
    out = pick_sentences(rows, exclude={"Cümle iki."}, n=10, seed=0)
    assert sorted(out) == ["Cümle bir.", "Cümle üç."]


def test_pick_sentences_drops_flagged_long_and_duplicate_text():
    base = {"duration": 4.0, "split": "train", "labelled": False}
    rows = [{**base, "normalized_text": "Aynı cümle."}, {**base, "normalized_text": "Aynı cümle."},
            {**base, "normalized_text": "Bayraklı.", "text_flags": ["digits"]},
            {**base, "normalized_text": "Uzun.", "duration": 21.0}]
    assert pick_sentences(rows, exclude=set(), n=10, seed=0) == ["Aynı cümle."]


def test_pick_sentences_is_seeded_and_capped():
    rows = [{"normalized_text": f"Cümle {i}.", "duration": 5.0, "split": "train", "labelled": False} for i in range(50)]
    a = pick_sentences(rows, exclude=set(), n=7, seed=3)
    assert len(a) == 7 and a == pick_sentences(rows, exclude=set(), n=7, seed=3)


def test_anchor_keep():
    ok = {"wer": 0.0, "read_aloud": [], "spk_cos": 0.82, "utmos": 3.1}
    assert anchor_keep(ok)
    assert not anchor_keep({**ok, "wer": 0.1}) and not anchor_keep({**ok, "spk_cos": 0.6})
    assert not anchor_keep({**ok, "read_aloud": ["öfkeyle"]}) and not anchor_keep({**ok, "utmos": None})
    assert not anchor_keep({**ok, "spk_cos": None})
