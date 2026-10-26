from pathlib import Path

from ttsdata.subtitles import Cue, clean_cue, load_segments, merge_continuations, parse


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def test_parse_srt_with_bom_crlf_and_glued_cue(tmp_path):
    body = (
        "﻿1\r\n00:00:01,000 --> 00:00:02,000\r\nMerhaba.\r\n\r\n"
        "2\r\n00:00:03,000 --> 00:00:04,000\r\nNasılsın?\r\n"          # no blank line before the next index
        "3\r\n00:00:05,000 --> 00:00:06,000\r\nİyiyim.\r\n"
    )
    cues = parse(_write(tmp_path, "a.srt", body))
    assert [c.lines for c in cues] == [["Merhaba."], ["Nasılsın?"], ["İyiyim."]]
    assert cues[1].start == 3.0 and cues[1].end == 4.0


def test_parse_vtt_header_and_position_tags(tmp_path):
    body = "WEBVTT\nKind: captions\nLanguage: tr\n\n00:00:20.335 --> 00:00:23.335 align:start position:0%\n(Jenerik müziği)\n\n00:01:00.000 --> 00:01:01.500\nGel.\n"
    cues = parse(_write(tmp_path, "a.vtt", body))
    assert len(cues) == 2 and cues[0].end == 23.335 and cues[1].lines == ["Gel."]


def test_clean_drops_sound_descriptions_and_lyrics():
    assert clean_cue(Cue(0, 1, ["(Siren çalıyor)", "(Duygusal müzik devam ediyor)"])) is None
    assert clean_cue(Cue(0, 1, ['("Güler Tacer - Unutamam Seni"', "devam ediyor)"])) is None
    assert clean_cue(Cue(0, 1, ["♪ Sevdik sevdalandık ♪"])) is None
    assert clean_cue(Cue(0, 1, ['"Sevdik sevdalandık', 'kördüğümle bağlandık"'])) is None


def test_clean_speaker_label_and_inline_parenthetical():
    s = clean_cue(Cue(0, 1, ["(Usta) Gaz ver.", "(Gaz veriyor)"]))
    assert s.text == "Gaz ver." and s.speaker_hint == "Usta" and s.flags == []
    s = clean_cue(Cue(0, 1, ["Gel (gülüyor) buraya."]))
    assert s.text == "Gel buraya."


def test_clean_flags_multi_speaker():
    assert "multi_speaker" in clean_cue(Cue(0, 1, ["-Gel.", "-Geliyorum."])).flags
    assert "multi_speaker" in clean_cue(Cue(0, 1, ["(Usta) Gel.", "(Çırak) Geliyorum."])).flags
    assert "multi_speaker" in clean_cue(Cue(0, 1, ["(Hep bir ağızdan) Evet!"])).flags
    s = clean_cue(Cue(0, 1, ["-Gel buraya."]))
    assert s.text == "Gel buraya." and s.flags == []


def test_merge_ellipsis_continuation_and_fragments():
    segs = [clean_cue(Cue(0.0, 2.0, ["Belki ölümden..."])), clean_cue(Cue(2.3, 4.0, ["...korkmadığı için."])),
            clean_cue(Cue(9.0, 10.0, ["Nice canlar..."])), clean_cue(Cue(12.0, 13.0, ["...nice cengaverler."]))]
    out = merge_continuations(segs, max_gap=0.6, max_gap_comma=0.3, max_dur=15)
    assert out[0].text == "Belki ölümden korkmadığı için." and out[0].n_cues == 2 and out[0].end == 4.0
    assert "fragment_end" in out[1].flags
    assert out[2].text == "nice cengaverler." and "fragment_start" in out[2].flags


def test_load_segments_counts(tmp_path):
    body = "1\n00:00:01,000 --> 00:00:02,000\n(Müzik)\n\n2\n00:00:03,000 --> 00:00:04,000\nMerhaba.\n"
    segs, st = load_segments(_write(tmp_path, "a.srt", body), 0.6, 0.3, 15)
    assert st == {"cues": 2, "speech_cues": 1, "segments": 1, "non_speech_cue": 1} and segs[0].text == "Merhaba."
