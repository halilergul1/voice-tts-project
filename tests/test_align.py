from ttsdata.asr import align, evaluate, speech_cover, trim_bounds
from ttsdata.config import Config


def W(words, t0=0.0, step=0.4):
    return [{"word": w, "start": round(t0 + i * step, 2), "end": round(t0 + (i + 1) * step - 0.05, 2), "prob": 0.9}
            for i, w in enumerate(words)]


def test_align_skips_bleed_both_sides():
    j0, j1, n, first, last = align("gel itiraf et".split(), W("görürler gel itiraf et senin için".split()))
    assert (j0, j1, n, first, last) == (1, 3, 3, True, True)


def test_align_fuzzy_and_split_words():
    j0, j1, n, _, last = align("niçin murad efendi niçin".split(), W("ne için murat efendi ne için".split()))
    assert (j0, j1) == (1, 5) and last
    j0, j1, n, _, _ = align("sizi mahkum etmek".split(), W("zordur sizi mahkumet".split()))
    assert (j0, j1) == (1, 2)


def test_align_none_and_truncated():
    assert align(["bulamıyorum"], W(["allahım"])) is None
    j0, j1, n, first, last = align("sultanım çok üzgün görünüyor".split(), W("sultanım çok üzgün".split()))
    assert (j0, j1, first, last) == (0, 2, True, False)


def test_evaluate_trims_to_span_and_scores_on_span():
    cfg = Config()
    rec = {"text": "Gel, itiraf et.", "window_dur": 4.0}
    asr = {"asr_text": "görürler Gel itiraf et senin için", "words": W("görürler gel itiraf et senin için".split(), 0.5, 0.5),
           "no_speech_prob": 0.0, "avg_logprob": -0.2}
    r = evaluate(rec, asr, cfg)
    assert r["cer"] == 0.0 and r["bleed_before"] == 1 and r["bleed_after"] == 2
    assert r["trim_start"] == 0.95 and r["trim_end"] == 2.5  # bounded by the neighbouring words


def test_edge_unmatched_keeps_gap():
    cfg = Config()
    t0, t1 = trim_bounds(1.0, 2.0, prev_end=0.3, next_start=3.0, first_ok=False, last_ok=True, window_dur=4.0, cfg=cfg)
    assert t0 == 0.3 and t1 == 2.0 + cfg.trim_margin_after


def test_speech_cover_relative_to_cue():
    r = {"cue_start": 10.0, "cue_end": 11.0, "window_dur": 3.0, "speech_ratio": 0.3}
    assert abs(speech_cover(r) - 0.9) < 1e-9


def test_snap_edges_caps_leading_and_trailing_silence():
    from ttsdata.build import snap_edges
    cfg = Config()
    a, b = snap_edges([(0.6, 1.5), (1.7, 2.4)], dur=3.0, cfg=cfg)
    assert abs(a - 0.5) < 1e-9 and abs(b - 2.6) < 1e-9
    a, b = snap_edges([(0.1, 2.9)], dur=3.0, cfg=cfg)
    assert a == 0.0 and b == 3.0
    assert snap_edges([], 2.0, cfg) == (0.0, 2.0)
