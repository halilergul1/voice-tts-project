from ttsdata.eventclips import accept, candidates, extension
from ttsdata.labels import EVENT_TAG, tag_text


def clip(cid="ep_00010", s=100.0, e=104.0, uid="ep", root="dataset", hint=None, members=None, text="Hadi!"):
    d = {"id": cid, "uid": uid, "root": root, "orig_start": s, "orig_end": e, "duration": round(e - s, 3),
         "speaker_hint": hint, "text": text, "normalized_text": text, "speaker_id": "ep_spk00",
         "speaker_cluster_size": 30, "split": "train", "series": "S", "source": "trt", "episode": 1, "cer": 0.0,
         "flags": [], "sample_rate": 24000}
    if members:
        d["members"] = members
    return d


def ev(tag="laugh", where="before", s=98.0, e=99.5, who=None, who_match=None, desc="Gülme sesi"):
    return {"tag": tag, "where": where, "start": s, "end": e, "who": who, "who_match": who_match, "desc": desc,
            "t": None, "dt": None, "char": None, "in_clip": False, "manner": None}


def test_extension_covers_the_event_at_the_matching_edge():
    assert extension(clip(), ev(), pad=0.2) == (97.8, 104.0, "start")
    assert extension(clip(), ev(where="after", s=104.6, e=105.5), pad=0.2) == (100.0, 105.7, "end")
    assert extension(clip(), ev(where="lead", s=99.0, e=104.2), pad=0.2) == (98.8, 104.0, "start")
    assert extension(clip(s=0.1, e=3.0), ev(s=0.0, e=0.5), pad=0.2) == (0.0, 3.0, "start")  # never before the episode start


def test_extension_refuses_unsafe_cases():
    assert extension(clip(), ev(who_match=False)) is None                       # the subtitle names someone else
    assert extension(clip(s=100.0, e=114.5), ev(s=98.0, e=99.5), max_dur=15.0) is None  # would exceed 15 s
    assert extension(clip(), ev(where="inside", s=101.0, e=102.0)) is None       # position unknown within the clip
    assert extension(clip(), ev(tag="snore")) is None                            # not a speech-adjacent sound we tag
    assert extension(clip(), ev(), neighbours=[(97.0, 98.4)]) is None            # another line inside the extension
    assert extension(clip(), ev(), neighbours=[(97.0, 97.7)]) is not None        # ends before the extension starts
    assert extension(clip(), ev(where="after", s=104.6, e=105.5), neighbours=[(105.6, 108.0)]) is None


def test_tag_text_places_the_turkish_tag():
    assert tag_text("Hadi!", "laugh", "start") == "[gülme] Hadi!"
    assert tag_text("Hadi!", "cry", "end") == "Hadi! [ağlama]"
    assert tag_text("Gel.", "sigh", "start") == "[iç çekme] Gel."
    assert EVENT_TAG["breath"] == "[nefes]"


def test_candidates_one_row_per_usable_event():
    base = clip()
    tags = {"ep_00010": {"id": "ep_00010", "style": [], "flags": [], "events": [
        ev(), ev(tag="cry", where="after", s=104.6, e=105.5), ev(who_match=False, s=96.0, e=97.0),
        ev(tag="laugh", where="inside", s=101.0, e=102.0)]}}
    rows = candidates([base], tags, neighbours={"ep": [(90.0, 95.0), (106.0, 110.0)]}, pad=0.2, max_dur=15.0)
    assert [r["id"] for r in rows] == ["ep_00010_laugh", "ep_00010_cry"]
    r = rows[0]
    assert (r["orig_start"], r["orig_end"], r["edge"], r["tag"]) == (97.8, 104.0, "start", "laugh")
    assert r["text"] == "Hadi!" and r["normalized_text"] == "Hadi!"  # plain: the audit / tier tools read these
    assert r["text_tagged"] == "[gülme] Hadi!" and r["normalized_text_tagged"] == "[gülme] Hadi!"
    assert r["base"] == "ep_00010" and r["root"] == "dataset" and r["verified"] is False
    assert r["event"]["desc"] == "Gülme sesi" and r["duration"] == 6.2
    assert rows[1]["text_tagged"] == "Hadi! [ağlama]" and rows[1]["edge"] == "end"


def test_candidates_skip_neighbours_that_are_the_merged_clips_own_members():
    merged = clip(cid="ep_00010_x2", s=100.0, e=106.0, root="dataset_long", members=["ep_00010", "ep_00011"])
    tags = {"ep_00010_x2": {"id": "ep_00010_x2", "style": [], "flags": [], "events": [ev()]}}
    neighbours = {"ep": [(100.0, 102.5, "ep_00010"), (103.0, 106.0, "ep_00011"), (97.0, 98.4, "ep_00009")]}
    assert candidates([merged], tags, neighbours=neighbours) == []  # ep_00009 sits inside the extension
    neighbours["ep"] = [(100.0, 102.5, "ep_00010"), (103.0, 106.0, "ep_00011")]
    assert len(candidates([merged], tags, neighbours=neighbours)) == 1


def test_accept_requires_alignment_cer_and_a_quiet_extension():
    ok = {"trim_start": 0.4, "cer": 0.03, "bleed_before": 0, "bleed_after": 0}
    assert accept(ok, "start", max_cer=0.10, max_bleed=1) is None
    assert accept({**ok, "trim_start": None}, "start", 0.10, 1) == "unaligned"
    assert accept({**ok, "cer": 0.2}, "start", 0.10, 1) == "high_cer"
    assert accept({**ok, "bleed_before": 2}, "start", 0.10, 1) == "words_in_extension"
    assert accept({**ok, "bleed_before": 2}, "end", 0.10, 1) is None       # the other edge is not the extension
    assert accept({**ok, "bleed_after": 1}, "end", 0.10, 1) is None        # one stray token ("ha") is tolerated
    assert accept({**ok, "bleed_after": 2}, "end", 0.10, 1) == "words_in_extension"
