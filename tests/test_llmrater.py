import base64
import json

from ttsdata.llmrater import (LABELS, PROMPT, backoff_seconds, parse_response, preds_for_scoring, report,
                              request_body)


def _resp(text, finish="STOP"):
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": finish}],
            "usageMetadata": {"promptTokenCount": 300, "candidatesTokenCount": 20, "totalTokenCount": 320}}


def test_request_body_sends_audio_and_the_prompt_only():
    body = request_body(b"RIFFfake", "audio/wav")
    parts = body["contents"][0]["parts"]
    assert [p for p in parts if "text" in p] == [{"text": PROMPT}]            # no transcript, nothing else
    audio = [p["inline_data"] for p in parts if "inline_data" in p]
    assert audio == [{"mime_type": "audio/wav", "data": base64.b64encode(b"RIFFfake").decode()}]
    gc = body["generationConfig"]
    assert gc["responseMimeType"] == "application/json" and gc["temperature"] == 0
    assert gc["responseSchema"]["properties"]["emotion"]["enum"] == list(LABELS)
    assert set(gc["responseSchema"]["required"]) == {"emotion", "confidence", "style"}
    assert "model" not in body                                                 # the model is in the URL


def test_parse_response_reads_json_even_when_fenced_or_odd():
    out = parse_response(_resp('{"emotion": "angry", "confidence": 0.8, "style": "shouting"}'))
    assert {k: out[k] for k in ("label", "conf", "style", "finish", "tokens")} == \
        {"label": "angry", "conf": 0.8, "style": "shouting", "finish": "STOP", "tokens": 320}
    out = parse_response(_resp('```json\n{"emotion": "Sad", "confidence": "0.55", "style": "tearful"}\n```'))
    assert out["label"] == "sad" and out["conf"] == 0.55
    out = parse_response(_resp('{"emotion": "fearful", "confidence": 1.7}'))
    assert out["label"] == "other" and out["conf"] == 1.0 and out["style"] == ""   # unknown class -> other, conf clamped
    out = parse_response({"candidates": [{"finishReason": "SAFETY"}]})
    assert out["label"] is None and out["finish"] == "SAFETY"
    out = parse_response(_resp("I cannot tell."))
    assert out["label"] is None and out["raw"] == "I cannot tell."


def test_preds_for_scoring_treats_other_and_failures_as_abstentions():
    rows = [{"id": "a", "label": "angry", "conf": 0.9}, {"id": "b", "label": "other", "conf": 0.7},
            {"id": "c", "label": None, "conf": None}, {"id": "d", "label": "neutral", "conf": 0.6}]
    assert preds_for_scoring(rows) == {"a": ("angry", 0.9), "b": (None, 0.7), "c": (None, 0.0), "d": ("neutral", 0.6)}


def test_backoff_grows_and_honours_retry_after():
    assert backoff_seconds(0) == 5 and backoff_seconds(1) == 10 and backoff_seconds(4) == 60   # capped
    assert backoff_seconds(0, retry_after=42.0) == 42.0


def test_report_compares_the_llm_with_emotion2vec():
    llm = [{"id": "a", "label": "angry", "conf": 0.9, "style": "shouting"}, {"id": "b", "label": "happy", "conf": 0.8, "style": "bright"},
           {"id": "c", "label": "other", "conf": 0.5, "style": "two voices"}]
    e2v = {"a": ("angry", 0.95), "b": ("neutral", 0.7), "c": ("sad", 0.65)}
    ratings = {"a": "angry", "b": "neutral", "c": "sad"}
    text = report("gemini-x", llm, e2v, ratings, cost_tokens=960)
    assert "# LLM rater: gemini-x" in text and "| angry | 1 | 1 | 1.0 |" in text
    assert "emotion2vec+" in text and "agree" in text.lower() and "shouting" in text
    json.dumps(preds_for_scoring(llm))


def test_pacer_spaces_request_starts_across_threads():
    from ttsdata.llmrater import Pacer
    now = [100.0]
    slept = []
    p = Pacer(rpm=30, clock=lambda: now[0], sleep=lambda s: (slept.append(s), now.__setitem__(0, now[0] + s)))
    assert p.wait() == 0.0 and p.wait() == 2.0 and p.wait() == 2.0      # 30 rpm -> one start every 2 s
    now[0] += 60
    assert p.wait() == 0.0                                                 # a long gap owes nothing
    assert slept == [2.0, 2.0]


def test_pacer_adapts_to_throttling():
    from ttsdata.llmrater import Pacer
    p = Pacer(rpm=30, clock=lambda: 0.0, sleep=lambda s: None, max_gap=10.0)
    assert p.slow_down() == 4.0 and p.slow_down() == 8.0 and p.slow_down() == 10.0   # doubles, capped
    for _ in range(40):
        p.recover()
    assert p.gap == 2.0                                                                # back to the configured rate


def test_quota_kind_reads_the_429_details():
    from ttsdata.llmrater import quota_kind
    day = {"error": {"details": [{"violations": [{"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                                                   "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}
    minute = {"error": {"details": [{"violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]}]}}
    assert quota_kind(day) == "day" and quota_kind(minute) == "minute" and quota_kind({"error": {}}) == "unknown"
    assert quota_kind({"error": {"message": "Quota exceeded for ... per day ..."}}) == "day"
