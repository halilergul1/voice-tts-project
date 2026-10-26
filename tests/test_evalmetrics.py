import json

from ttsdata.evalmetrics import lora_config_from_checkpoint, summarize


def test_summarize_aggregates_scores():
    s = [{"wer": 0.0, "cer": 0.0, "utmos": 4.0, "spk_cos": 0.9, "rtf": 0.5},
         {"wer": 0.5, "cer": 0.2, "utmos": 3.0, "spk_cos": 0.7, "rtf": 1.5}]
    out = summarize(s)
    assert out["n"] == 2 and abs(out["wer"] - 0.25) < 1e-9 and abs(out["cer"] - 0.1) < 1e-9
    assert abs(out["utmos_mean"] - 3.5) < 1e-9 and abs(out["utmos_p10"] - 3.1) < 1e-9
    assert abs(out["spk_consistency"] - 0.8) < 1e-9 and abs(out["rtf"] - 1.0) < 1e-9


def test_summarize_tolerates_missing_fields():
    out = summarize([{"wer": 0.1, "cer": 0.05}, {"wer": 0.3, "cer": 0.1, "utmos": 3.0}])
    assert out["n"] == 2 and abs(out["wer"] - 0.2) < 1e-9 and abs(out["utmos_mean"] - 3.0) < 1e-9
    assert out["spk_consistency"] is None and out["rtf"] is None
    assert summarize([])["n"] == 0


def test_lora_config_from_checkpoint_filters_to_known_fields(tmp_path):
    (tmp_path / "lora_config.json").write_text(json.dumps({"r": 64, "alpha": 64, "dropout": 0.0, "enable_lm": True,
                                                           "enable_dit": True, "enable_proj": False, "extra": 1}))
    cfg = lora_config_from_checkpoint(tmp_path, ["r", "alpha", "dropout", "enable_lm", "enable_dit", "enable_proj"])
    assert cfg == {"r": 64, "alpha": 64, "dropout": 0.0, "enable_lm": True, "enable_dit": True, "enable_proj": False}
    assert lora_config_from_checkpoint(tmp_path / "missing", ["r"]) is None
