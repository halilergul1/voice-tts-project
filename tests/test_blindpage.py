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
