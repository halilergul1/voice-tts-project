from ttsdata.eventverify import TAG_CLASSES, SPEECH_CLASSES, evidence, laugh_syllables, speech_tokens, verdict


def test_speech_tokens_ignores_laughter_interjections_and_hallucinations():
    assert speech_tokens("Hahahaha") == []
    assert speech_tokens("Ha ha ha ha.") == []
    assert speech_tokens("Haha") == []
    assert speech_tokens("hıh hmm ah of oh aa ay") == []
    assert speech_tokens("(gülme) [kahkaha] (ağlama)") == []
    assert speech_tokens("Altyazı M.K.") == []                      # Whisper's credits hallucination on non-speech
    assert speech_tokens("Ahahahah Altyazı M.K.") == []              # the hallucination is dropped, not the whole text
    assert speech_tokens("Gel buraya Altyazı M.K.") == ["gel", "buraya"]
    assert speech_tokens("k m .K.") == []                              # consonant-only fragments are never words
    assert speech_tokens("Altyazı: Ahmet Yılmaz") == []
    assert speech_tokens("") == []
    assert speech_tokens("İzlediğiniz için teşekkür ederim.") == ["izlediğiniz", "için", "teşekkür", "ederim"]
    assert speech_tokens("Asya üzgün haliyle Ömer'in yanına") == ["asya", "üzgün", "haliyle", "ömer'in", "yanına"]
    assert speech_tokens("hıh ne oldu") == ["ne", "oldu"]
    assert speech_tokens("Kıhıhıhıhıhıhı") == []                     # a cough / laugh transcribed with a k
    assert speech_tokens("Ahem.") == [] and speech_tokens("Öhö öhö") == []
    assert speech_tokens("Eeeh aaaah aaaah") == []


def test_tag_classes_cover_every_extendable_tag():
    from ttsdata.labels import EXTENDABLE_TAGS
    for tag in EXTENDABLE_TAGS:
        assert TAG_CLASSES.get(tag), tag
    assert "Laughter" in TAG_CLASSES["laugh"] and "Crying, sobbing" in TAG_CLASSES["cry"]
    assert "Speech" in SPEECH_CLASSES and "Narration, monologue" in SPEECH_CLASSES


def test_verdict_rules():
    thr = {"speech_prob": 0.3, "tag_prob": 0.2}
    # the narrator: words in the extension and the tagger hears speech
    assert verdict("breath", ["izlediğiniz", "için"], speech_prob=0.8, tag_prob=0.1, thr=thr) == (False, "speech_in_extension")
    # words but no speech according to the tagger, and the laugh is there: Whisper hallucinated
    assert verdict("laugh", ["altyazı", "mk"], speech_prob=0.05, tag_prob=0.6, thr=thr) == (True, None)
    # clean extension, laugh detected
    assert verdict("laugh", [], speech_prob=0.1, tag_prob=0.5, thr=thr) == (True, None)
    # clean extension (no words, no speech-like audio), nothing positively detected: accepted as "clean" by default,
    # because listening found the subtitle's sound present in every such sample; switchable off
    assert verdict("laugh", [], speech_prob=0.1, tag_prob=0.05, thr=thr) == (True, None)
    assert verdict("laugh", [], speech_prob=0.1, tag_prob=0.05, thr={**thr, "accept_clean": False}) == (False, "unconfirmed")
    assert evidence("laugh", [], speech_prob=0.1, tag_prob=0.05, thr=thr) == "clean"
    assert evidence("laugh", [], speech_prob=0.1, tag_prob=0.5, thr=thr) == "tagger"
    assert evidence("laugh", [], speech_prob=0.1, tag_prob=0.0, thr=thr, laugh_syl=3) == "laugh_syllables"
    assert evidence("cry", [], speech_prob=0.5, tag_prob=0.05, thr=thr) is None   # speech-like audio, no words: unconfirmed
    # speech according to the tagger alone (no words) is not enough to reject a laugh that is detected
    assert verdict("laugh", [], speech_prob=0.5, tag_prob=0.5, thr=thr) == (True, None)
    # but speech without the tagged sound is rejected
    assert verdict("cry", [], speech_prob=0.5, tag_prob=0.05, thr=thr) == (False, "unconfirmed")
    # per-tag override
    assert verdict("breath", [], speech_prob=0.1, tag_prob=0.12, thr={**thr, "tag_prob_breath": 0.1}) == (True, None)
    # a whole sentence from Whisper is speech whatever the tagger says (the narrator once scored 0.23)
    assert verdict("shout", ["atıf", "tabancasını", "çıkarırken"], speech_prob=0.23, tag_prob=0.5, thr=thr) == (False, "speech_in_extension")
    assert verdict("laugh", ["evet"], speech_prob=0.23, tag_prob=0.5, thr=thr) == (True, None)
    # Whisper's laughter syllables confirm a laugh even when the tagger is quiet
    assert verdict("laugh", [], speech_prob=0.1, tag_prob=0.0, thr=thr, laugh_syl=4) == (True, None)
    assert verdict("laugh", [], speech_prob=0.1, tag_prob=0.0, thr={**thr, "accept_clean": False}, laugh_syl=1) == (False, "unconfirmed")
    assert verdict("cry", [], speech_prob=0.1, tag_prob=0.0, thr={**thr, "accept_clean": False}, laugh_syl=4) == (False, "unconfirmed")


def test_laugh_syllables_counts_ha_like_tokens():
    assert laugh_syllables("Hahahaha") == 4
    assert laugh_syllables("Ha ha ha ha.") == 4
    assert laugh_syllables("Hı hı hı") == 3
    assert laugh_syllables("Kıhıhıhı") == 4
    assert laugh_syllables("Ahahahah") == 4
    assert laugh_syllables("Haaa!") == 1
    assert laugh_syllables("İzlediğiniz için teşekkür ederim.") == 0
    assert laugh_syllables("Altyazı M.K.") == 0
    assert laugh_syllables("Ahahahahahahah Altyazı M.K.") >= 4         # a laugh followed by the credits hallucination


def test_pad_to_centres_short_audio():
    import numpy as np
    from ttsdata.eventverify import pad_to
    y = np.ones(400, dtype=np.float32)
    out = pad_to(y, sr=1000, min_sec=1.0)
    assert len(out) == 1000 and out[:300].sum() == 0 and out[300:700].sum() == 400 and out[700:].sum() == 0
    assert pad_to(np.ones(1500, dtype=np.float32), sr=1000, min_sec=1.0).shape == (1500,)
