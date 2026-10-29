import numpy as np

from ttsdata.voicegender import DESCRIPTION, DESCRIPTIONS, clip_f0, describe, speaker_gender


def test_speaker_gender_labels_clear_cases_only():
    assert speaker_gender([110, 120, 130, 115])["gender"] == "male"
    assert speaker_gender([210, 220, 190, 230])["gender"] == "female"
    assert speaker_gender([158, 160, 162, 170])["gender"] is None  # median in the gap
    assert speaker_gender([110, 230, 120, 220])["gender"] is None  # clips disagree: mixed cluster
    assert speaker_gender([110, 120])["gender"] is None  # too few clips
    assert speaker_gender([110, None, 120, 130])["n"] == 3  # unvoiced clips are ignored
    assert set(DESCRIPTION) == {"male", "female"}


def test_clip_f0_finds_the_pitch_of_a_tone():
    sr = 16000
    t = np.arange(sr) / sr
    y = 0.3 * np.sin(2 * np.pi * 120 * t).astype(np.float32)
    assert abs(clip_f0(y, sr) - 120) < 5
    assert clip_f0(np.zeros(sr, dtype=np.float32), sr) is None


def test_describe_prefixes_known_genders_and_leaves_styled_text():
    assert describe("Merhaba.", "female") == "(bir kadın sesi) Merhaba."
    assert describe("Merhaba.", "male", DESCRIPTIONS["en"]) == "(A man's voice) Merhaba."
    assert describe("Merhaba.", None) == "Merhaba."
    assert describe("(neşeyle) Merhaba.", "male") == "(neşeyle) Merhaba."
