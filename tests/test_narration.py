from ttsdata.narration import (clean_span_text, link_lines, name_clusters, narration_gaps, parse_description,
                               spans_from_words, speech_chunks)


def test_parse_description():
    assert parse_description("Şivenaz zindanda sinirle dört dönerken") == [{"name": "Şivenaz", "emotion": "angry", "word": "sinirle"}]
    assert parse_description("Ayşe'nin saçlarını okşarken ağlıyor.")[0]["name"] == "Ayşe"
    assert parse_description("Sade Nur sevinçle gülümsedi.")[0] == {"name": "Sade Nur", "emotion": "happy", "word": "sevinçle"}
    assert parse_description("Ağlamamak için dudağını ısırdı.") == []
    assert parse_description("Kapıyı açıp içeri girdi.") == []


def test_parse_description_negation_and_positive_verb_forms():
    assert parse_description("Ali gülümsemeden baktı.") == []
    assert parse_description("Ali korkmadan yürüdü.") == []
    assert parse_description("Ali ağlamıyor.") == []
    # positive forms that also start with -ma/-me after the stem
    assert parse_description("Ali ağlamaya başladı.")[0]["emotion"] == "sad"
    assert parse_description("Ali ağlamak üzere.")[0]["emotion"] == "sad"


def test_parse_description_crying_class_and_one_row_per_emotion():
    # "ağlayarak" also starts with the sad stem "ağla": crying while talking is its own class
    assert parse_description("Mert ağlayarak konuşuyor.") == [{"name": "Mert", "emotion": "crying", "word": "ağlayarak"}]
    # two happy words in one description give one row
    assert len(parse_description("Sade Nur sevinçle gülümsedi.")) == 1
    # a name in an earlier sentence does not carry over; Turkish capital İ lowercases to i
    assert parse_description("Ali kapıyı açtı. İçeride sinirle bekliyor.") == [{"name": "İçeride", "emotion": "angry", "word": "sinirle"}]
    assert parse_description("HİDDETLE bağırdı") == [{"name": None, "emotion": "angry", "word": "HİDDETLE"}]


def test_name_clusters_majority():
    rows = [{"series": "S", "speaker_hint": "Ayşe", "speaker_id": "S_spk1"}] * 8 + [{"series": "S", "speaker_hint": "Ayşe", "speaker_id": "S_spk2"}] * 2
    assert name_clusters(rows) == {("S", "Ayşe"): "S_spk1"}
    assert name_clusters(rows[:4]) == {}                                  # too few named lines
    split = [{"series": "S", "speaker_hint": "Ali", "speaker_id": f"S_spk{k % 2}"} for k in range(10)]
    assert name_clusters(split) == {}                                     # no cluster holds 60 %


def test_link_lines_window_and_cluster():
    desc = {"name": "Ayşe", "end": 100.0}
    lines = [{"id": "a", "start": 99.0, "speaker_id": "S_spk1"}, {"id": "b", "start": 110.0, "speaker_id": "S_spk2"},
             {"id": "c", "start": 115.0, "speaker_id": "S_spk1"}, {"id": "d", "start": 125.0, "speaker_id": "S_spk1"}]
    assert link_lines(desc, lines, "S_spk1") == ["a", "c"]
    assert link_lines({"name": None, "end": 100.0}, lines, None) == []     # no line starts within 5 s after
    # the same character's next description ends the window
    assert link_lines({"name": "Ayşe", "end": 100.0, "next": 112.0}, lines, "S_spk1") == ["a"]
    assert link_lines({"name": None, "end": 108.0}, lines, None) == ["b"]  # first line within 5 s, any speaker


def test_narration_gaps_between_speech_cues():
    cues = [(10.0, 12.0), (12.5, 14.0), (20.0, 22.0), (21.0, 23.0)]
    # gaps shorter than min_gap after the edge padding go; overlapping cues merge; head and tail of the episode count
    assert narration_gaps(cues, total=30.0, min_gap=1.5, pad=0.2) == [(0.0, 9.8), (14.2, 19.8), (23.2, 30.0)]
    assert narration_gaps([], total=5.0) == [(0.0, 5.0)]


def test_spans_from_words_splits_on_pauses_and_drops_junk():
    words = [{"word": "Ayşe", "start": 1.0, "end": 1.4}, {"word": "sinirle", "start": 1.5, "end": 2.0},
             {"word": "bakıyor.", "start": 2.1, "end": 2.5},
             {"word": "Altyazı", "start": 6.0, "end": 6.5}, {"word": "M.K.", "start": 6.6, "end": 7.0},
             {"word": "Kapı", "start": 9.0, "end": 9.3}, {"word": "açılıyor.", "start": 9.4, "end": 9.9}]
    assert spans_from_words(words, max_pause=1.5) == [
        {"start": 1.0, "end": 2.5, "text": "Ayşe sinirle bakıyor."},
        {"start": 9.0, "end": 9.9, "text": "Kapı açılıyor."}]


def test_parse_description_skips_lookalike_words_and_names():
    assert parse_description("Hemen arkasından Hasan balkon korkuluklarının yanına geçiyor.") == []  # railing, not fear
    assert parse_description("Ali, Sevinç'e sarıldı.") == []          # a capitalised word inside a sentence is a name
    assert parse_description("Sinirle kapıyı çarptı.")[0]["emotion"] == "angry"  # sentence-initial capital still counts


def test_spans_from_words_drops_the_describers_intro():
    words = [{"word": "Bu", "start": 0.0, "end": 0.2}, {"word": "dizinin", "start": 0.3, "end": 0.6},
             {"word": "betimlemesi", "start": 0.7, "end": 1.2},
             {"word": "TRT", "start": 5.0, "end": 5.3}, {"word": "tarafından", "start": 5.4, "end": 5.9},
             {"word": "Sesli", "start": 6.0, "end": 6.3}, {"word": "Betimleme", "start": 6.4, "end": 6.9},
             {"word": "Derneğine", "start": 7.0, "end": 7.5}, {"word": "yaptırılmıştır.", "start": 7.6, "end": 8.2}]
    assert spans_from_words(words) == []


def test_parse_description_present_tense_of_smile():
    # "gülümse-" narrows to "gülümsü-" before -yor: the most common form in the descriptions
    assert parse_description("Ayşe gülümsüyor.") == [{"name": "Ayşe", "emotion": "happy", "word": "gülümsüyor"}]
    assert parse_description("Ayşe gülümseyerek bakıyor.")[0]["emotion"] == "happy"


def test_speech_chunks_merge_short_pauses_and_cap_length():
    # Seamless drops words on long inputs, so speech is fed in chunks of at most max_len seconds
    speech = [(0.0, 2.0), (2.5, 4.0), (6.0, 7.0), (7.2, 20.0)]
    assert speech_chunks(speech, max_pause=1.0, max_len=10.0) == [(0.0, 4.0), (6.0, 7.0), (7.2, 13.6), (13.6, 20.0)]
    assert speech_chunks([(5.0, 6.0), (0.0, 1.0)], max_pause=1.0, max_len=10.0) == [(0.0, 1.0), (5.0, 6.0)]
    assert speech_chunks([]) == []


def test_clean_span_text_drops_hallucinations_and_the_describers_credit():
    assert clean_span_text(" Farah odaya  girdi. ") == "Farah odaya girdi."
    assert clean_span_text("Altyazı M.K.") is None
    assert clean_span_text("Bu dizinin betimlemesi TRT tarafından Sesli Betimleme Derneğine yaptırılmıştır.") is None
    assert clean_span_text("...") is None


def test_clean_span_text_drops_bracketed_sound_captions():
    # Seamless writes subtitle-style captions in brackets or stars; "(Korkma)" would otherwise count as fear
    assert clean_span_text("(Gülme sesi)") is None
    assert clean_span_text("(Çeviri:Tamam)") is None
    assert clean_span_text("Ali kapıyı açtı. (Korkma) *gülme sesleri*") == "Ali kapıyı açtı."
