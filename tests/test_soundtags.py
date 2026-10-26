from ttsdata.soundtags import Event, attach, classify, count_unique_events, hint_style, mine_cues
from ttsdata.subtitles import Cue


def test_classify_vocal_events_with_who():
    d = classify("Gülme sesi")
    assert (d.kind, d.tag, d.who) == ("vocal", "laugh", None)
    assert (classify("Adile gülme sesi").tag, classify("Adile gülme sesi").who) == ("laugh", "Adile")
    assert classify("Gülme sesleri").tag == "laugh"
    assert classify("Cevahir kahkaha atıyor").tag == "laugh"
    assert classify("Ağlama sesi").tag == "cry"
    assert classify("Kadın ağlıyor").tag == "cry"
    assert classify("Nefes sesi").tag == "breath"
    assert classify("Derin nefes alıp veriyor").tag == "breath"
    assert (classify("Eşref iç çekti").tag, classify("Eşref iç çekti").who) == ("sigh", "Eşref")
    assert classify("Öksürme sesi").tag == "cough"
    assert classify("Afra çığlık atıyor").tag == "scream"
    assert classify("Fehmi bağırıyor").tag == "shout"
    assert classify("Acıyla inledi").tag == "moan"
    assert classify("Esneme sesi").tag == "yawn"
    d = classify("Korkarak bağırıyor")
    assert (d.kind, d.tag, d.manner) == ("vocal", "shout", "scared")


def test_classify_manner_background_and_noise():
    assert (classify("Ağlayarak").kind, classify("Ağlayarak").tag) == ("manner", "crying")
    assert classify("Fısıldayarak konuşuyorlar").tag == "whispering"
    assert classify("Fısıltıyla").tag == "whispering"
    assert classify("Karadeniz ağzıyla").tag == "dialect"
    d = classify("Fonda kahkaha")
    assert (d.kind, d.tag) == ("background", "laugh")
    assert classify("Gerilim müziği devam ediyor").kind == "music"
    assert classify("Hüzünlü müzik").kind == "music"  # an emotion word inside a music cue is not a speech manner
    assert classify('Keman eşliğinde "Yine bir gülnihal" çalmaya devam ediyor').kind == "music"
    assert classify("Ağıt- Ayşenur Kolivar- Yistambilako").kind == "music"
    assert classify("Telefon çalıyor").kind == "sfx"
    assert classify("Kapı açılma sesi").kind == "sfx"
    assert classify("Üst üste konuşmalar").kind == "overlap"
    assert classify("***").kind == "censored"
    assert classify("Sessizlik").kind == "silence"


def test_classify_names_are_not_tags():
    for name in ("Gülden", "Gülenay", "Ayşegül", "Badegül", "Neşe", "Nisan ses", "Erkek dış ses", "İngiliz Sefir"):
        assert classify(name).kind == "other", name


def test_hint_style_splits_name_and_manner():
    assert hint_style("Afra ses ağlayarak") == ("Afra ses", ["crying"], [])
    assert hint_style("Ömer ağlayarak") == ("Ömer", ["crying"], [])
    assert hint_style("Korkarak bağırıyor") == (None, ["scared", "shouting"], [])
    assert hint_style("Gülme sesi") == (None, [], [])        # a sound before the line, not how the line is spoken
    assert hint_style("Afra ağlıyor") == ("Afra", ["crying"], [])  # a finite verb on the line describes its delivery
    assert hint_style("Mutlu") == ("Mutlu", [], [])            # a single capitalised word can be a name
    assert hint_style("Nefes") == ("Nefes", [], [])
    assert hint_style("Nisan ses") == ("Nisan ses", [], [])
    assert hint_style("***") == (None, [], ["censored"])
    assert hint_style(None) == (None, [], [])


def test_mine_cues_positions():
    cues = [Cue(10.0, 11.0, ["(Gülme sesi)"]),
            Cue(12.0, 14.0, ["(Ağlayarak) Neden gittin?"]),
            Cue(15.0, 17.0, ["Sen (***) bir adamsın."]),
            Cue(18.0, 19.0, ["(Gürdal) Gel buraya."]),
            Cue(20.0, 22.0, ["(Gerilim müziği)"])]
    ev = mine_cues(cues)
    got = [(e.start, e.kind, e.tag, e.position) for e in ev]
    assert (10.0, "vocal", "laugh", "standalone") in got
    assert (12.0, "manner", "crying", "lead") in got
    assert (15.0, "censored", None, "inline") in got
    assert (20.0, "music", None, "standalone") in got
    assert not any(e.start == 18.0 for e in ev)  # a speaker name is not a sound tag


def E(s, e, kind, tag=None, position="standalone", who=None, manner=None, desc="x", nominal=False):
    return Event(s, e, desc, kind, tag, who, manner, position, nominal)


def test_attach_style_flags_and_events_around_a_clip():
    clip = {"id": "c", "orig_start": 100.0, "orig_end": 104.0, "speaker_hint": None}
    events = [E(100.2, 103.8, "manner", "crying", position="lead"),   # the clip's own cue: style
              E(101.5, 102.5, "vocal", "laugh"),                    # inside: audio is in the clip
              E(98.0, 99.5, "vocal", "sigh", who="Nisan"),          # 0.5 s before
              E(104.6, 105.5, "vocal", "cough"),                    # 0.6 s after
              E(110.0, 111.0, "vocal", "cry"),                      # too far
              E(99.0, 106.0, "music"),
              E(103.0, 104.5, "background", "laugh"),
              E(100.5, 101.0, "censored", position="inline")]
    t = attach(clip, events, adjacent=1.5)
    assert t["style"] == ["crying"]
    assert t["flags"] == ["bg_laugh", "censored", "music"]
    where = {(x["tag"], x["where"]) for x in t["events"]}
    assert where == {("laugh", "inside"), ("sigh", "before"), ("cough", "after")}
    inside = next(x for x in t["events"] if x["where"] == "inside")
    assert inside["in_clip"] is True and abs(inside["t"] - 1.5) < 1e-9
    before = next(x for x in t["events"] if x["where"] == "before")
    assert before["in_clip"] is False and abs(before["dt"] - 0.5) < 1e-9 and before["who"] == "Nisan"
    assert (before["start"], before["end"]) == (98.0, 99.5)  # episode time, for re-cutting the clip to include it


def test_attach_places_gap_events_in_merged_clips():
    clip = {"id": "m", "orig_start": 10.0, "orig_end": 16.0, "speaker_hint": "Nisan",
            "text": "Gel. Hadi gidelim.", "member_spans": [[10.0, 12.0], [12.6, 16.0]], "member_text_ends": [4, 18]}
    t = attach(clip, [E(12.1, 12.5, "vocal", "laugh", who="Nisan"), E(13.0, 13.4, "vocal", "breath", who="Faruk")],
               adjacent=1.5)
    gap = next(x for x in t["events"] if x["tag"] == "laugh")
    assert gap["where"] == "gap:0" and gap["char"] == 4 and gap["in_clip"] is True and gap["who_match"] is True
    other = next(x for x in t["events"] if x["tag"] == "breath")
    assert other["where"] == "inside" and other["who_match"] is False


def test_attach_ignores_a_lead_manner_from_a_neighbouring_cue():
    clip = {"id": "c", "orig_start": 100.0, "orig_end": 102.0, "speaker_hint": "Ömer ağlayarak"}
    t = attach(clip, [E(101.8, 104.0, "manner", "whispering", position="lead")], adjacent=1.5)
    assert t["style"] == ["crying"]  # from the hint; the next cue's manner overlaps too little


def test_attach_events_grazing_a_clip_edge_count_as_before_or_after():
    clip = {"id": "c", "orig_start": 100.0, "orig_end": 104.0, "speaker_hint": None}
    t = attach(clip, [E(99.0, 100.1, "vocal", "laugh"), E(103.95, 105.0, "vocal", "cough")], adjacent=1.5)
    got = {(x["tag"], x["where"], x["dt"]) for x in t["events"]}
    assert got == {("laugh", "before", 0.0), ("cough", "after", 0.0)}


def test_classify_exclusion_categories_and_extra_vocals():
    assert classify("Almanca konuşma").kind == "foreign"
    assert classify("Fransızca konuşuyor").kind == "foreign"
    assert classify("Yavaşça konuşuyor").kind != "foreign"
    assert classify("Anlaşılmayan konuşma").kind == "unintelligible"
    assert classify("Gürdal'ın konuşması anlaşılmıyor").kind == "unintelligible"
    assert classify("Anons kadın ses").kind == "channel"
    assert classify("TV erkek sesi").kind == "channel"
    assert classify("Telefon çalıyor").kind == "sfx"  # a ringing phone is not a phone voice
    assert classify("Ezan okunuyor").kind == "music"
    assert (classify("Gürdal horluyor").tag, classify("Gürdal horluyor").who) == ("snore", "Gürdal")
    assert classify("Boğazını temizledi").tag == "throat_clear"
    assert classify("İnliyorlar").tag == "moan"


def test_attach_turns_exclusion_events_into_flags():
    clip = {"id": "c", "orig_start": 0.0, "orig_end": 3.0, "speaker_hint": None}
    t = attach(clip, [E(0.5, 2.0, "foreign"), E(1.0, 2.0, "unintelligible"), E(0.0, 3.0, "channel", position="lead")])
    assert t["flags"] == ["channel", "foreign_speech", "unintelligible"]


def test_classify_lead_names_are_not_sounds():
    for name in ("Recep", "Deniz", "Yağmur", "Ateş", "Vural", "Gökhan", "Masal", "Ezgi", "Melodi", "Nefes", "Mutlu"):
        assert classify(name, position="lead").kind == "other", name
    assert classify("Recep").kind == "other"
    assert classify("Telefon çalıyor", position="lead").kind == "sfx"
    assert classify("Ağlayarak", position="lead").kind == "manner"
    assert classify("Ağlıyor", position="lead").kind == "vocal"
    assert classify("Kahkaha").kind == "vocal"
    assert classify("Yağmur sesi").kind == "sfx"
    for voice in ("Erkek sesi", "Kadın sesi", "Nisan'ın sesi"):
        assert classify(voice).kind == "other", voice
    assert classify("TV erkek sesi").kind == "channel"


def test_classify_bleep_and_phone_conversation():
    assert classify("Bip").kind == "censored"
    assert classify("Nisan telefonda konuşuyor").kind != "channel"
    assert classify("Telefondan").kind == "channel"


def test_classify_marks_nominal_sound_forms():
    assert classify("Öksürme sesi").nominal and classify("Gülme sesi").nominal and classify("Kahkaha").nominal
    assert not classify("Ağlıyor").nominal and not classify("Fehmi bağırıyor").nominal


def test_attach_lead_sound_is_an_event_not_a_style():
    clip = {"id": "c", "orig_start": 100.5, "orig_end": 101.5, "speaker_hint": None}
    t = attach(clip, [E(100.0, 102.0, "vocal", "cough", position="lead", desc="Öksürme sesi", nominal=True)])
    assert t["style"] == [] and [(x["tag"], x["where"], x["in_clip"]) for x in t["events"]] == [("cough", "lead", False)]
    t = attach(clip, [E(100.0, 102.0, "vocal", "cry", position="lead", desc="Ağlıyor")])
    assert t["style"] == ["crying"] and t["events"] == []


def test_attach_merged_clip_style_needs_every_member():
    clip = {"id": "m", "orig_start": 10.0, "orig_end": 15.0, "speaker_hint": None, "text": "Bir. İki.",
            "member_spans": [[10.0, 12.0], [12.5, 15.0]], "member_text_ends": [4, 9], "member_hints": [None, None]}
    one = attach(clip, [E(10.0, 12.0, "manner", "crying", position="lead")])
    assert one["style"] == [] and one["style_members"] == [["crying"], []]
    both = attach(clip, [E(10.0, 12.0, "manner", "crying", position="lead"),
                         E(12.5, 15.0, "manner", "crying", position="lead")])
    assert both["style"] == ["crying"]
    hinted = attach({**clip, "member_hints": ["Ömer ağlayarak", "Ömer ağlayarak"]}, [])
    assert hinted["style"] == ["crying"]


def test_attach_rounds_times_to_subtitle_precision():
    clip = {"id": "c", "orig_start": 100.0, "orig_end": 104.0, "speaker_hint": None}
    t = attach(clip, [E(101.537, 102.5, "vocal", "laugh"), E(104.731, 105.2, "vocal", "cough")])
    got = {x["tag"]: (x["t"], x["dt"]) for x in t["events"]}
    assert got == {"laugh": (1.5, None), "cough": (None, 0.7)}


def test_count_unique_events_uses_the_episode_not_the_clip_id():
    ev = {"tag": "laugh", "in_clip": False, "start": 50.0}
    tags = [{"id": "ep_00009_x2", "events": [ev]}, {"id": "ep_00012_x2", "events": [ev]}]
    uid_of = {"ep_00009_x2": "ep", "ep_00012_x2": "ep"}
    assert count_unique_events(tags, uid_of) == {("laugh", "adjacent"): 1}


def test_discrete_sounds_are_events_in_any_verb_form():
    for d in ("Öksürüyor", "Boğazını temizledi", "Esniyor", "Horluyor", "Islık çalıyor"):
        assert classify(d, position="lead").nominal, d
    assert hint_style("Öksürüyor") == (None, [], [])
    assert not classify("Gülüyor", position="lead").nominal  # laughing while talking is a delivery
