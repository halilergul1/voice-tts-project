from ttsdata.textvote import vote


def test_vote_second_recogniser_backs_the_subtitle():  # Whisper misheard
    assert vote("Marşa bas.", "marşabaz", "Marşa bas.") == "subtitle"
    assert vote("386, 387", "bunu verecek", "üç yüz seksen altı üç yüz seksen yedi") == "subtitle"  # digits expanded


def test_vote_second_recogniser_backs_whisper():  # the subtitle cleaned up what was said
    assert vote("Bütün paketi bitirsin.", "çekedi bitirsin ya ulan", "Çekedi bitirsin ya ulan!") == "spoken"


def test_vote_drops_when_nobody_agrees():
    assert vote("Hadi güzelim hadi ya.", "gelmiyor mu", "hadi gel bakalım") == "drop"
    assert vote("Merhaba.", "merhaba", "") == "drop"


def test_vote_prefers_the_subtitle_when_all_agree():
    assert vote("Tamam abi.", "tamam abi", "tamam abi") == "subtitle"
