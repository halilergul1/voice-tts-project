import numpy as np

from ttsdata.separation import choose_source, relative_loudness_db


def test_relative_loudness_db_is_instrumental_minus_vocals():
    v = np.full(24000, 0.1, dtype=np.float32)
    assert abs(relative_loudness_db(v, v * 0.1) - (-20.0)) < 0.1
    assert relative_loudness_db(v, np.zeros_like(v)) <= -80.0
    assert relative_loudness_db(np.zeros_like(v), np.zeros_like(v)) <= -80.0  # both silent: treat as no music


def test_choose_source_threshold():
    # the separator leaves a -12..-25 dB "instrumental" residue even on clean read speech, so the default sits above that
    assert choose_source(-35.0) == "original"
    assert choose_source(-12.0) == "original"
    assert choose_source(-8.0) == "separated"
    assert choose_source(-10.0) == "separated"
    assert choose_source(-10.0, threshold_db=-5.0) == "original"


def test_choose_source_keeps_original_when_the_stem_is_unreliable():
    assert choose_source(5.0) == "original"  # voice ended up in the instrumental stem
    assert choose_source(-5.0, vocal_ratio=0.1) == "original"  # vocal stem lost most of the energy
    assert choose_source(-5.0, vocal_ratio=0.8) == "separated"
