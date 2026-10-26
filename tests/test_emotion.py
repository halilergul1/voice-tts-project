from ttsdata.emotion import (COARSE, agree, calibration_sample, coarse_label, parse_ratings, score_ratings,
                             suggest_thresholds)


def S(**kw):
    base = {"angry": 0.0, "disgusted": 0.0, "fearful": 0.0, "happy": 0.0, "neutral": 0.0, "other": 0.0, "sad": 0.0,
            "surprised": 0.0, "unknown": 0.0}
    base.update(kw)
    return base


def test_coarse_label_needs_confidence_and_margin():
    assert coarse_label(S(angry=0.8, neutral=0.1), min_conf=0.6, min_margin=0.2) == ("angry", 0.8)
    assert coarse_label(S(angry=0.5, neutral=0.4), 0.6, 0.2) == (None, 0.5)        # not confident
    assert coarse_label(S(angry=0.65, sad=0.5), 0.6, 0.2) == (None, 0.65)          # two coarse classes close
    assert coarse_label(S(neutral=0.9), 0.6, 0.2) == ("neutral", 0.9)
    assert coarse_label(S(fearful=0.9, angry=0.05), 0.6, 0.2) == (None, 0.9)       # fine classes are not labelled
    assert coarse_label(S(other=0.7, happy=0.25), 0.6, 0.2) == (None, 0.7)
    assert set(COARSE) == {"angry", "happy", "sad", "neutral"}


def test_agree_combines_two_scorers():
    assert agree("angry", "angry") == "angry"
    assert agree("angry", "sad") is None
    assert agree("angry", None) is None
    assert agree("angry", "missing") == "angry"       # second scorer not available: the first decides
    assert agree(None, "missing") is None


def test_calibration_sample_is_stratified_and_deterministic():
    rows = []
    for series in ("A", "B"):
        for lab in ("angry", "sad", "happy", "neutral", None):
            for k in range(30):
                rows.append({"id": f"{series}_{lab}_{k}", "series": series, "coarse": lab, "conf": 0.9 - k * 0.01})
    picked = calibration_sample(rows, n=50, seed=1)
    assert len(picked) == 50 and len(set(picked)) == 50
    by_lab = {}
    for cid in picked:
        by_lab.setdefault(cid.split("_")[1], 0)
        by_lab[cid.split("_")[1]] += 1
    assert by_lab == {"angry": 10, "sad": 10, "happy": 10, "neutral": 10, "None": 10}
    assert sum(1 for c in picked if c.startswith("A_")) == 25
    assert picked == calibration_sample(rows, n=50, seed=1)
    assert len(calibration_sample(rows[:7], n=50, seed=1)) == 7


def test_parse_and_score_ratings():
    preds = {"a": ("angry", 0.9), "b": ("angry", 0.7), "c": ("sad", 0.8), "d": (None, 0.5), "e": ("happy", 0.95)}
    text = "id,rating\na,angry\nb,neutral\nc,sad\nd,sad\ne,happy\nf,\n"
    ratings = parse_ratings(text)
    assert ratings == {"a": "angry", "b": "neutral", "c": "sad", "d": "sad", "e": "happy"}
    m = score_ratings(preds, ratings)
    assert m["n"] == 5 and abs(m["accuracy"] - 0.75) < 1e-9          # of the 4 labelled: a, c, e right, b wrong
    assert m["per_class"]["angry"] == {"predicted": 2, "correct": 1, "precision": 0.5, "rated": 1, "recall": 1.0}
    assert m["per_class"]["sad"]["recall"] == 0.5                       # d was sad but unlabelled
    assert m["confusion"][("angry", "neutral")] == 1


def test_suggest_thresholds_per_class():
    preds = {f"a{i}": ("angry", 0.5 + i * 0.1) for i in range(5)}      # 0.5 .. 0.9
    ratings = {"a0": "neutral", "a1": "neutral", "a2": "angry", "a3": "angry", "a4": "angry"}
    thr = suggest_thresholds(preds, ratings, target_precision=0.76)
    assert abs(thr["angry"] - 0.7) < 1e-9                              # at >= 0.7 all three are right
    assert suggest_thresholds({"x": ("sad", 0.9)}, {"x": "angry"}, 0.76) == {"sad": None}  # never reaches target


def test_coarse_from_eiv_maps_the_fine_dimensions():
    from ttsdata.emotion_eiv import coarse_from_eiv
    assert coarse_from_eiv({"Anger": 2.5, "Sadness": 0.3, "Amusement": 0.2, "Contentment": 0.1}) == "angry"
    assert coarse_from_eiv({"Anger": 0.2, "Sadness": 2.1, "Distress": 1.0, "Amusement": 0.2}) == "sad"
    assert coarse_from_eiv({"Anger": 0.1, "Sadness": 0.2, "Amusement": 1.9, "Elation": 1.2}) == "happy"
    assert coarse_from_eiv({"Anger": 0.4, "Sadness": 0.3, "Amusement": 0.2}) == "neutral"   # nothing above 1.0 on 0-4
    assert coarse_from_eiv({"Anger": 1.6, "Sadness": 1.5}) is None                          # two emotions close
    assert coarse_from_eiv({}) is None


def test_coarse_from_eiv_accepts_file_style_dimension_names():
    from ttsdata.emotion_eiv import coarse_from_eiv
    assert coarse_from_eiv({"Pleasure_Ecstasy": 2.0, "Anger": 0.1}) == "happy"
    assert coarse_from_eiv({"Impatience_and_Irritability": 1.8, "Sadness": 0.2}) == "angry"


def test_parse_e2v_output_normalises_labels():
    from ttsdata.emotion import parse_e2v
    labels = ['生气/angry', '厌恶/disgusted', '恐惧/fearful', '开心/happy', '中立/neutral', '其他/other', '难过/sad', '吃惊/surprised', '<unk>']
    scores = [0.006, 0.034, 0.005, 0.343, 0.134, 0.0, 0.447, 0.031, 0.0]
    out = parse_e2v(labels, scores)
    assert set(out) == {"angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"}
    assert out["sad"] == 0.447 and out["unknown"] == 0.0
