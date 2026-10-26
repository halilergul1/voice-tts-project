import numpy as np

from ttsdata.quality import episode_outliers, silence_edges, text_flags


def test_text_flags_catches_unreadable_text():
    assert "digits" in text_flags("Saat 3'te.", "Saat 3'te.")
    assert text_flags("Saat üçte.", "Saat üçte.") == []
    assert "abbrev" in text_flags("Dr. Ahmet geldi.", "Dr. Ahmet geldi.")
    assert "allcaps" in text_flags("TBMM toplandı.", "TBMM toplandı.")
    assert "leftover" in text_flags("(Gülme) Hadi.", "(Gülme) Hadi.")
    assert "leftover" in text_flags("♪ Gel ♪", "♪ Gel ♪")
    assert "fragment" in text_flags("...gelmedi.", "...gelmedi.")
    assert "latin" in text_flags("Welcome to Istanbul.", "Welcome to Istanbul.")


def test_text_flags_ignores_ordinary_turkish():
    assert text_flags("Eşref abi, İzzet'in telefonu. Babası arıyor.", "Eşref abi, İzzet'in telefonu. Babası arıyor.") == []
    assert text_flags("Ne oldu? Söyle!", "Ne oldu? Söyle!") == []
    assert text_flags("Ben de inanmadım zaten.", "Ben de inanmadım zaten.") == []  # sentence-final "." is not an abbreviation
    # a short word ending a sentence, followed by a capitalised word, is not an abbreviation
    assert text_flags("Yeter artık ya. Vallahi yeter.", "Yeter artık ya. Vallahi yeter.") == []
    assert text_flags("Biraz geciktim de. Biraz.", "Biraz geciktim de. Biraz.") == []
    assert "abbrev" in text_flags("A. Kadir geldi.", "A. Kadir geldi.")  # an initial is
    assert "abbrev" in text_flags("ünv. kampüsünde", "ünv. kampüsünde")  # period then lowercase is
    assert text_flags("Üç yıl. on sekiz yaşına kadar.", "Üç yıl. on sekiz yaşına kadar.") == []  # num2words output
    assert text_flags("Not al Danyal.", "Not al Danyal.") == []  # "not" is a Turkish word


def test_silence_edges_measures_leading_and_trailing_silence():
    sr = 24000
    y = np.concatenate([np.zeros(int(0.3 * sr)), np.random.default_rng(0).normal(0, 0.1, sr), np.zeros(int(0.6 * sr))]).astype(np.float32)
    lead, trail = silence_edges(y, sr)
    assert abs(lead - 0.3) < 0.03 and abs(trail - 0.6) < 0.03


def test_silence_edges_all_silent_clip():
    lead, trail = silence_edges(np.zeros(24000, dtype=np.float32), 24000)
    assert lead == 1.0 and trail == 1.0


def test_episode_outliers_flags_bad_cer_and_keep_rate():
    rows = [{"uid": "good", "cer": 0.01} for _ in range(50)] + [{"uid": "bad", "cer": 0.15} for _ in range(50)]
    rows += [{"uid": f"ok{i}", "cer": 0.01 + 0.001 * i} for i in range(5) for _ in range(20)]
    out = episode_outliers(rows)
    assert [o["uid"] for o in out] == ["bad"] and "mean_cer" in out[0] and out[0]["n"] == 50
