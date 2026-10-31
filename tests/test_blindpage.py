from ttsdata.blindpage import ab_key, parse_answers, style_items, tally_ab, tally_style


def test_ab_key_is_seeded_and_mixed():
    k = ab_key(20, seed=6)
    assert k == ab_key(20, seed=6) and set(k) == {"A", "B"} and len(k) == 20


def test_tally_ab():
    assert tally_ab(["A", "B", "A"], ["A", "A", "same"]) == {"ours": 1, "theirs": 1, "same": 1}
    assert tally_ab(["A", "B"], ["A", "-"]) == {"ours": 1, "theirs": 0, "same": 0, "unanswered": 1}


def test_style_items_and_tally():
    by = {s: [f"{s}/{i:04d}.wav" for i in range(10)] for s in ("angry", "sad", "neutral")}
    items = style_items(by, n_per=4, seed=0)
    assert len(items) == 12 and {i["style"] for i in items} == {"angry", "sad", "neutral"}
    idx = [i["wav"].split("/")[1] for i in items]
    assert all(a != b for a, b in zip(idx, idx[1:]))
    t = tally_style(items, [i["style"] for i in items])
    assert t["per_style"]["angry"] == {"correct": 4, "n": 4}
    assert t["confusion"][("sad", "sad")] == 4


def test_parse_answers():
    assert parse_answers("style check: 1:angry 2:- 3:sad") == {1: "angry", 2: "-", 3: "sad"}
    assert parse_answers("1:cry 10:none") == {1: "cry", 10: "none"}


def test_ab_layout_side_not_tied_to_sentence():  # review: one seed drove both the shuffle and the sides
    from ttsdata.blindpage import ab_layout

    lay = ab_layout(20, seed=6)
    assert sorted(j for j, _ in lay) == list(range(20)) and sorted(s for _, s in lay) == ["A"] * 10 + ["B"] * 10
    sides_of_0 = {dict(ab_layout(20, seed=s))[0] for s in range(10)}
    first_half = [{dict(ab_layout(20, seed=s))[j] for j in range(10)} for s in range(10)]
    assert sides_of_0 == {"A", "B"} and any(len(h) == 2 for h in first_half)


def test_storage_key_changes_with_the_page_key():  # review: a rebuilt page must not restore old picks
    from ttsdata.blindpage import storage_key

    assert storage_key("p", {"a": 1}) == storage_key("p", {"a": 1})
    assert storage_key("p", {"a": 1}) != storage_key("p", {"a": 2})


def test_style_spec_and_item_text():  # review: tag pages printed the tag; one sentence file for every style
    from ttsdata.blindpage import item_text, parse_style_spec

    assert parse_style_spec("laugh=data/eval/r/wavs@data/eval/tags_laugh.txt") == (
        "laugh", "data/eval/r/wavs", "data/eval/tags_laugh.txt")
    assert parse_style_spec("none=data/eval/p/wavs") == ("none", "data/eval/p/wavs", None)
    sents = ["[gülme] Hadi gidelim.", "(öfkeyle) Neden geldin?"]
    assert item_text("x/0000.wav", sents) == "Hadi gidelim." and item_text("x/0001.wav", sents) == "Neden geldin?"
    assert item_text("x/0007.wav", sents) == "" and item_text("x/abc.wav", sents) == ""
