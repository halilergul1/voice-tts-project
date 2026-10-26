from ttsdata.labels import STRONG_STYLES, STYLE_PREFIX, allowed_labels, labelled_text, oversample


def T(style=(), events=()):
    return {"style": list(style), "flags": [], "events": list(events)}


def gap_event(tag="laugh", char=4, who_match=None, where="gap:0"):
    return {"tag": tag, "where": where, "char": char, "who_match": who_match, "in_clip": where != "before"}


def test_labelled_text_adds_a_turkish_style_prefix():
    assert labelled_text("Neden gittin?", T(style=["crying"]), {"crying"}, set()) == "(ağlayarak) Neden gittin?"
    two = labelled_text("Git!", T(style=["shouting", "crying"]), {"crying", "shouting"}, set())
    assert two == "(ağlayarak, bağırarak) Git!"  # table order, not input order
    assert labelled_text("Git!", T(style=["crying"]), set(), set()) == "Git!"  # style not frequent enough: no label
    assert labelled_text("Git!", None, {"crying"}, {"laugh"}) == "Git!"
    assert labelled_text("Git!", T(style=["dialect"]), {"dialect"}, set()) == "Git!"  # no prefix defined for it


def test_labelled_text_inserts_gap_tags_only_when_safe():
    tags = T(events=[gap_event()])
    assert labelled_text("Gel. Hadi gidelim.", tags, set(), {"laugh"}) == "Gel. [gülme] Hadi gidelim."
    assert labelled_text("Gel. Hadi gidelim.", T(events=[gap_event(who_match=False)]), set(), {"laugh"}) == "Gel. Hadi gidelim."
    assert labelled_text("Gel. Hadi gidelim.", T(events=[gap_event(where="inside")]), set(), {"laugh"}) == "Gel. Hadi gidelim."
    assert labelled_text("Gel. Hadi gidelim.", tags, set(), set()) == "Gel. Hadi gidelim."  # tag not frequent enough
    both = labelled_text("Gel. Hadi gidelim.", T(style=["crying"], events=[gap_event()]), {"crying"}, {"laugh"})
    assert both == "(ağlayarak) Gel. [gülme] Hadi gidelim."
    two_gaps = T(events=[gap_event(char=4), gap_event(tag="sigh", char=10, where="gap:1")])
    assert labelled_text("Gel. Hadi. Gidelim.", two_gaps, set(), {"laugh", "sigh"}) == "Gel. [gülme] Hadi. [iç çekme] Gidelim."


def test_allowed_labels_thresholds():
    styles = ["crying"] * 50 + ["shouting"] * 49
    tags = ["laugh"] * 300 + ["cry"] * 299
    assert allowed_labels(styles, tags, min_style=50, min_tag=300) == ({"crying"}, {"laugh"})
    assert allowed_labels([], [], 50, 300) == (set(), set())


def test_strong_styles_are_the_ones_stage_1_must_not_train_as_plain():
    assert {"crying", "screaming", "shouting", "laughing", "pained"} <= STRONG_STYLES
    assert "dialect" not in STRONG_STYLES and all(s in STYLE_PREFIX for s in STRONG_STYLES)


def test_oversample_repeats_labelled_rows_under_a_cap():
    rows = [{"id": f"p{i}", "labelled": False} for i in range(90)] + [{"id": f"l{i}", "labelled": True} for i in range(10)]
    out = oversample(rows, repeat=3, max_share=0.4, seed=0)
    assert sum(1 for r in out if r["labelled"]) == 30 and sum(1 for r in out if not r["labelled"]) == 90
    out = oversample(rows, repeat=30, max_share=0.4, seed=0)   # 10 labelled x 30 = 300 would be 77 %: capped
    n_lab = sum(1 for r in out if r["labelled"])
    assert n_lab <= 0.4 * len(out) + 1e-9 and n_lab >= 50
    assert [r["id"] for r in oversample(rows, repeat=1, max_share=0.4, seed=0)] == [r["id"] for r in rows]


def test_auto_styles_from_emotion_labels():
    from ttsdata.labels import auto_styles
    rows = [{"id": "a", "coarse": "angry", "conf": 0.9}, {"id": "b", "coarse": "neutral", "conf": 0.95},
            {"id": "c", "coarse": "sad", "conf": 0.5}, {"id": "d", "coarse": None, "conf": 0.9}]
    assert auto_styles(rows, min_conf=0.6) == {"a": ["angry"]}                    # neutral is the absence of a prefix
    assert auto_styles(rows, min_conf=0.6, per_class={"sad": 0.4}) == {"a": ["angry"], "c": ["sad"]}
    assert auto_styles(rows, min_conf=0.95) == {}
