from ttsdata.stylemetrics import read_aloud, scoring_texts, strip_labels, style_hit, style_summary, tag_realised


def test_strip_labels():
    assert strip_labels("(öfkeyle) Neden geldin?") == "Neden geldin?"
    assert strip_labels("[gülme] Hadi gidelim. [nefes]") == "Hadi gidelim."
    assert strip_labels("Saat beşte (belki) gelir.") == "Saat beşte (belki) gelir."  # only a leading prefix


def test_scoring_texts_ignore_laughter_only_when_tagged():
    assert scoring_texts("[gülme] Hadi gidelim.", "Hahaha hadi gidelim") == ("hadi gidelim", "hadi gidelim")
    assert scoring_texts("Hadi gidelim.", "Hahaha hadi gidelim") == ("hadi gidelim", "hahaha hadi gidelim")


def test_read_aloud():
    assert read_aloud("öfkeyle neden geldin", "(öfkeyle) Neden geldin?") == ["öfkeyle"]
    assert read_aloud("gülme hadi", "[gülme] Hadi.") == ["gülme"]
    assert read_aloud("gülme sakın", "Gülme sakın.") == []  # the word is in the sentence itself
    assert read_aloud("angry neden", "Neden?", extra_words=("angry",)) == ["angry"]


def test_style_hit_and_summary():
    p_ang = {"angry": 0.6, "neutral": 0.3, "sad": 0.1, "happy": 0.0, "fearful": 0.9}  # non-coarse classes ignored
    assert style_hit(p_ang, "angry") and not style_hit(p_ang, "sad")
    rows = [{"e2v": p_ang}, {"e2v": {"angry": 0.1, "neutral": 0.8, "sad": 0.1, "happy": 0.0}}]
    s = style_summary(rows, "angry")
    assert s["n"] == 2 and s["hit_rate"] == 0.5 and s["neutral_rate"] == 0.5 and abs(s["target_post"] - 0.35) < 1e-9


def test_tag_realised():
    assert tag_realised("Hahaha hadi gidelim", "laugh") is True
    assert tag_realised("hadi gidelim", "laugh") is False
    assert tag_realised("hadi gidelim", "cry") is None
