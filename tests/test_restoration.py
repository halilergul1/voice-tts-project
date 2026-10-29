import numpy as np

from ttsdata.restoration import (context_span, ending_drop_db, episode_audio, last_word_db, noise_pad, rms_match,
                                 trim_span)


def test_context_span_clamps_to_the_episode():
    assert context_span(10.0, 12.0, 0.5, 100.0) == (9.5, 12.5, 0.5)
    assert context_span(0.2, 1.0, 0.5, 100.0) == (0.0, 1.5, 0.2)
    t0, t1, lead = context_span(99.0, 99.8, 0.5, 100.0)
    assert (t0, t1, lead) == (98.5, 100.0, 0.5)


def test_trim_span_cuts_the_clip_back_out_of_the_context():
    sr = 100
    y = np.arange(500, dtype=np.float32)
    out = trim_span(y, sr, lead=0.5, dur=2.0)
    assert len(out) == 200 and out[0] == 50 and out[-1] == 249
    assert len(trim_span(y, sr, lead=4.0, dur=2.0)) == 100  # runs off the end: whatever is left


def _speech_then(sr, tail_gain, tail_s=0.4, body_s=2.0):
    t = np.arange(int(sr * (body_s + tail_s))) / sr
    y = 0.3 * np.sin(2 * np.pi * 200 * t).astype(np.float32)
    y[int(sr * body_s):] *= tail_gain
    return y


def test_last_word_db_and_ending_drop_detect_a_faded_last_word():
    sr = 16000
    full = _speech_then(sr, 1.0)
    faded = _speech_then(sr, 0.1)  # last 0.4 s 20 dB down, still above the -30 dB "voiced" floor
    assert abs(last_word_db(full, sr)) < 1.0
    assert ending_drop_db(faded, sr, full, sr) < -15
    assert abs(ending_drop_db(full, sr, full, sr)) < 0.5


def test_last_word_db_ignores_trailing_silence():
    sr = 16000
    y = np.concatenate([_speech_then(sr, 1.0), np.zeros(sr // 2, dtype=np.float32)])
    assert abs(last_word_db(y, sr)) < 1.0


def test_noise_pad_and_rms_match():
    rng = np.random.default_rng(0)
    x = np.ones(10, dtype=np.float32)
    p = noise_pad(x, 5, rng)
    assert len(p) == 20 and np.all(p[5:15] == 1) and 0 < np.abs(p[:5]).max() < 0.01
    y = rms_match(np.full(8, 0.5, dtype=np.float32), np.full(8, 0.1, dtype=np.float32))
    assert np.allclose(y, 0.1)


def test_episode_audio_skips_subtitle_files(tmp_path):
    (tmp_path / "audio.tr.vtt").write_text("WEBVTT")
    assert episode_audio(tmp_path) is None
    (tmp_path / "audio.webm").write_bytes(b"")
    assert episode_audio(tmp_path).name == "audio.webm"


def test_word_check_falls_back_only_when_sidon_loses_words():
    from ttsdata.restoration import word_check

    calls = []
    clean = lambda: calls.append(1) or 0.30  # noqa: E731  # the music-removed clip already misses words
    assert word_check(0.02, 0.0, clean) == ("sidon", None)  # close to the build's CER: no second transcription
    assert calls == []
    assert word_check(0.36, 0.0, clean) == ("fallback_words", 0.30)  # 0.06 worse than the clean clip
    assert word_check(0.33, 0.0, clean) == ("sidon", 0.30)  # within 0.05 of the clean clip
    assert word_check(0.12, None, lambda: 0.0) == ("fallback_words", 0.0)  # no build CER: treated as 0
