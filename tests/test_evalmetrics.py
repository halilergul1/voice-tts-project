import json

import pytest

from ttsdata.evalmetrics import lora_config_from_checkpoint, lora_part, parse_lora_scale, summarize


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


def test_lora_config_from_checkpoint_reads_the_trainers_nested_block(tmp_path):
    # VoxCPM's trainer writes {"base_model": ..., "lora_config": {...}}; reading only the top level gave an empty
    # config, so inference built no LoRA layers and skipped every weight (2026-10-28 smoke run)
    (tmp_path / "lora_config.json").write_text(json.dumps({"base_model": "/m/VoxCPM2", "lora_config": {
        "enable_lm": True, "enable_dit": True, "enable_proj": False, "r": 64, "alpha": 64, "dropout": 0.0,
        "target_modules_lm": ["q_proj"]}}))
    cfg = lora_config_from_checkpoint(tmp_path, ["r", "alpha", "dropout", "enable_lm", "enable_dit", "enable_proj",
                                                 "target_modules_lm"])
    assert cfg == {"enable_lm": True, "enable_dit": True, "enable_proj": False, "r": 64, "alpha": 64, "dropout": 0.0,
                   "target_modules_lm": ["q_proj"]}


def test_lora_part_splits_language_model_and_acoustic_decoder():
    assert lora_part("base_lm.layers.3.self_attn.q_proj") == "lm"
    assert lora_part("residual_lm.layers.0.self_attn.o_proj") == "lm"
    assert lora_part("feat_decoder.estimator.decoder.layers.1.self_attn.k_proj") == "dit"
    assert lora_part("feat_decoder.estimator._orig_mod.decoder.layers.1.self_attn.v_proj") == "dit"  # torch.compile
    assert lora_part("enc_to_lm_proj") == "other"


def test_parse_lora_scale():
    assert parse_lora_scale("1,0") == (1.0, 0.0)
    assert parse_lora_scale("0.5") == (0.5, 0.5)
    with pytest.raises(ValueError):
        parse_lora_scale("1,0,1")
